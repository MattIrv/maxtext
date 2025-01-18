"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Standalone checkpointer - only saves and restores checkpoints at regular intervals, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import time

from typing import Sequence
from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
from jax import numpy as jnp
import numpy as np
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import storage_utils

import checkpointing
import max_utils
import max_logging
import pyconfig
from train import setup_mesh_and_model, get_first_step, validate_train_config, save_checkpoint

from layers import models
import orbax.checkpoint

Transformer = models.Transformer

CHECKPOINT_RESTORE_TIME_DIRECTORY = "ckpt_restore_time"
CHECKPOINT_WRITE_TIME_DIRECTORY = "ckpt_write_time"

def checkpoint_loop(config, state=None):
  """Main Checkpointing loop.
  Saves checkpoints.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  ckpt_read_time = []
  ckpt_write_time = []
  init_rng, _, checkpoint_manager, mesh, model, _, tx = setup_mesh_and_model(config)

  unboxed_abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
  # A barrier to sync all hosts before starting to restore checkpoint
  jax.experimental.multihost_utils.sync_global_devices("Barrier before load")
  checkpoint_load_start = datetime.datetime.now()
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state, _ = checkpointing.load_state_if_possible(
        checkpoint_manager, None, config.load_parameters_path, config.load_full_state_path, unboxed_abstract_state
    )
    if state and not isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
      state = state["items"]

  jax.block_until_ready(state)
  checkpoint_load_end = datetime.datetime.now()
  
  if state is not None:  # Checkpoint was available for restore
    if jax.process_index() == 0:
      max_logging.log(f"STANDALONE CHECKPOINTER : Initial checkpoint restored in : {checkpoint_load_end - checkpoint_load_start}")
  else:  # Checkpoint was unavailable, state needs to be initialized
    state, _, _, _ = max_utils.setup_training_state(model, None, tx, config, init_rng, mesh, checkpoint_manager)
  state = add_entropy_to_checkpoint(state)

  start_step = get_first_step(state)  # this is the start_step for training
  for step in np.arange(start_step, config.steps):
    start_time = datetime.datetime.now()
    if checkpoint_manager is not None:
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("Barrier before save")
      start_time = datetime.datetime.now()
      if save_checkpoint(checkpoint_manager, int(step), state):
        checkpoint_manager.wait_until_finished()
        end_time = datetime.datetime.now()
        checkpoint_write_time = (end_time - start_time).total_seconds()
        ckpt_write_time.append([jax.process_index(), checkpoint_write_time])
        if jax.process_index() == 0:
          max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint saved in {end_time - start_time} ,step {step}, on host 0")
    if jax.process_index() == 0:
      elapsed_time = datetime.datetime.now() - start_time
      time_to_wait = config.per_step_interval - elapsed_time.total_seconds()
      if time_to_wait > 0:
        max_logging.log(f"Waiting {time_to_wait} seconds to reach step time of {config.per_step_interval} seconds for step {step}")
        time.sleep(time_to_wait)
    jax.experimental.multihost_utils.sync_global_devices("Barrier after step")
          
  if config.gcs_metrics_bucket:
    max_logging.log(f"Uploading write metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    
    base_name = f"{jax.process_index()}.csv"
    storage_utils.upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, CHECKPOINT_WRITE_TIME_DIRECTORY, base_name), 
        ckpt_write_time
    )
    max_logging.log(f"Finished uploading write metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}  for run {config.run_name}")

  for step in np.arange(start_step, config.steps):
    if checkpoint_manager is not None:
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("Barrier before restore")
      start_time = datetime.datetime.now()
      try:
        state = checkpoint_manager.restore(
              step,
              args=orbax.checkpoint.args.Composite(items=orbax.checkpoint.args.PyTreeRestore(item=unboxed_abstract_state)),
            )
        if state:
          state = state["items"]
      except FileNotFoundError:
        # No checkpoint was found for the step, presumably because one was not produced for the step. Continue on.
        continue
      jax.block_until_ready(state)
      end_time = datetime.datetime.now()
      checkpoint_restore_time = (end_time - start_time).total_seconds()
      ckpt_read_time.append([jax.process_index(), checkpoint_restore_time])
      if jax.process_index() == 0:
        max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint restored in {end_time - start_time} ,step {step}, on host 0")
  
  if config.gcs_metrics_bucket:
    max_logging.log(f"Uploading restore metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    
    base_name = f"{jax.process_index()}.csv"
    storage_utils.upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, CHECKPOINT_RESTORE_TIME_DIRECTORY, base_name), 
        ckpt_read_time
    )
    max_logging.log(f"Finished uploading restore metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()} for run {config.run_name}")

  return state


def add_entropy_to_checkpoint(state):
  """Introduce randomness in checkpoints. This is useful to simulate real checkpoints, without training.
  Args:
    state: Initial state
  Returns:
    state: Returns state with entropy added to the optimizer state.
  """
  with jax.spmd_mode("allow_all"):
    opt_0 = state.opt_state[0]
    opt_0 = opt_0._replace(mu=jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), state.params))
    opt_0 = opt_0._replace(nu=jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), state.params))
    new_opt = [opt_0] + list(state.opt_state[1:])
    state = state.replace(opt_state=new_opt)
    return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_cpu_enable_gloo_collectives", True)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_train_config(config)
  print(f"Found {jax.device_count()} devices.")
  print(f"Found {jax.process_count()} processes.")
  print(f"Found {jax.devices()} devices.")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  checkpoint_loop(config)


if __name__ == "__main__":
  app.run(main)
