"""Restore checkpoint auxiliaries without giving AdamW ownership of source tensors."""

import copy


def restore_optimizer(optimizer, state):
    optimizer.load_state_dict(copy.deepcopy(state))
