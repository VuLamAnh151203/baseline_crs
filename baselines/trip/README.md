# TRIP

Bidirectional target-constrained action/topic planning with forward/backward encoders, agreement
loss and constrained decoding. Qwen is used only for the shared realization/evaluation interface.

TRIP requires data preparation, separate dataset checkpoint training, measured development budget
selection, a selection lock and then test. Follow `tdrl/README_TRIP.md`; the staged procedure is
intentional because choosing the matched budget before measurement would invalidate the comparison.

TopDial uses declared weak path supervision because native intermediate TRIP paths are absent.
