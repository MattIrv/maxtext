#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Page Managers."""

from flax import struct
import jax.numpy as jnp

import common_types
import jax

Array = common_types.Array
DType = common_types.DType

AxisNames = common_types.AxisNames


@struct.dataclass
class PageState:
  page_status: Array
  seq_page_idx_mappings: Array
  seq_lengths: Array
  seq_num_pages: Array
  seq_page_indices: Array
  seq_page_slice_indices: Array


class PageManager:
  """Page Manager"""
  def __init__(
      self,
      num_pages: int,
      page_size: int,
      slots: int,
      max_target_length: int,
      max_prefill_predict_length: int,
  ) -> None:
    assert max_target_length % page_size == 0
    assert max_prefill_predict_length % page_size == 0
    max_pages_per_slot = max_target_length // page_size
    max_pages_per_prefill = max_prefill_predict_length // page_size
    page_state = PageState(
      page_status=jnp.zeros((num_pages,), dtype=jnp.int32),
      seq_page_idx_mappings=jnp.zeros((slots, max_pages_per_slot), dtype=jnp.int32),
      seq_lengths=jnp.zeros((slots,), dtype=jnp.int32),
      seq_num_pages=jnp.zeros((slots,), dtype=jnp.int32),
      seq_page_indices=jnp.zeros((slots,), dtype=jnp.int32),
      seq_page_slice_indices=jnp.zeros((slots,), dtype=jnp.int32),
    )
    self.num_pages = num_pages
    self.page_size = page_size
    self.slots = slots
    self.max_target_length = max_target_length
    self.max_prefill_predict_length = max_prefill_predict_length
    self.max_pages_per_slot = max_pages_per_slot
    self.max_pages_per_prefill = max_pages_per_prefill
    self.page_state = page_state

  def release_slot_pages(self, slot: int, page_state: PageState) -> PageState:
    """Release sequence slot and the pages assigned to the slot."""

    page_status = page_state.page_status
    seq_page_idx_mappings = page_state.seq_page_idx_mappings
    seq_lengths = page_state.seq_lengths
    seq_num_pages = page_state.seq_num_pages
    seq_page_indices = page_state.seq_page_indices
    seq_page_slice_indices = page_state.seq_page_slice_indices

    for i in range(seq_num_pages[slot]):
      page_idx = seq_page_idx_mappings[slot][i]
      page_status[page_idx] = 0
      seq_page_idx_mappings[slot] = 0

    seq_lengths[slot] = 0
    seq_num_pages[slot] = 0
    seq_page_indices[slot] = 0
    seq_page_slice_indices[slot] = 0

    return PageState(
      page_status,
      seq_page_idx_mappings,
      seq_lengths,
      seq_num_pages,
      seq_page_indices,
      seq_page_slice_indices
    )

  def reserve_prefill_slot_pages(self, slot: int, true_length: int, page_state: PageState) -> PageState:
    """Reserve pages for prefill slot."""

    page_state = self.release_slot_pages(slot, page_state)

    page_status = page_state.page_status
    seq_page_idx_mappings = page_state.seq_page_idx_mappings
    seq_lengths = page_state.seq_lengths
    seq_num_pages = page_state.seq_num_pages
    seq_page_indices = page_state.seq_page_indices
    seq_page_slice_indices = page_state.seq_page_slice_indices

    prefill_slot_num_pages = jnp.ceil(true_length / self.page_size).astype(jnp.int32)
    prefill_slot_page_slice_idx = (true_length - 1) % self.page_size

    for i in range(prefill_slot_num_pages):
      assert jnp.count_nonzero(page_status[1:]) != self.num_pages-1, "All pages are in use."
      page_idx = jnp.where((page_status[1:]==0), size=1)[0]
      page_status[page_idx] = 1
      seq_page_idx_mappings[slot][i]= page_idx

    seq_lengths[slot] = true_length
    seq_num_pages[slot] = prefill_slot_num_pages
    seq_page_indices[slot] = page_idx
    seq_page_slice_indices[slot] = prefill_slot_page_slice_idx

    return PageState(
      page_status,
      seq_page_idx_mappings,
      seq_lengths,
      seq_num_pages,
      seq_page_indices,
      seq_page_slice_indices
    )

  def reserve_decode_step_pages(self, page_state: PageState) -> PageState:
    """Reserve pages for decode step."""

    page_status = page_state.page_status
    seq_page_idx_mappings = page_state.seq_page_idx_mappings
    seq_lengths = page_state.seq_lengths
    seq_num_pages = page_state.seq_num_pages
    seq_page_indices = page_state.seq_page_indices
    seq_page_slice_indices = page_state.seq_page_slice_indices

    seq_lengths += jax.lax.cond(seq_lengths > 0, 1, 0)
    current_seq_num_pages = seq_num_pages
    seq_num_pages = jnp.ceil(seq_lengths / self.page_size).astype(jnp.int32)
    seq_page_slice_indices = (seq_lengths - 1) % self.page_size
    seq_new_page = seq_num_pages - current_seq_num_pages
    num_new_pages = jnp.count_nonzero(seq_new_page)
    if num_new_pages:
      updating_slots = jnp.where((seq_new_page>0), size=self.slots)
      for i in range(num_new_pages):
        assert jnp.count_nonzero(page_status[1:]) != self.num_pages-1, "All pages are in use."
        slot = updating_slots[i]
        page_idx = jnp.where((page_status[1:]==0), size=1)[0]
        page_status[page_idx] = 1
        seq_page_idx_mappings[slot][seq_num_pages[slot]-1] = page_idx
        seq_page_indices[slot] = page_idx

    return PageState(
      page_status,
      seq_page_idx_mappings,
      seq_lengths,
      seq_num_pages,
      seq_page_indices,
      seq_page_slice_indices
    )
