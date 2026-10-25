import pytest

from tdrl.difftod_durecdial_evaluation import verify_selected_policy
from tdrl.provenance import atomic_json


def test_durecdial_cannot_silently_load_a_topdial_fitting(tmp_path):
    from test_difftod_evaluation import completed_fit

    chosen, manifest = completed_fit(tmp_path)
    with pytest.raises(ValueError, match="completed native-budget fitting"):
        verify_selected_policy(tmp_path)
    manifest["stage"] = "DiffTOD-DuRecDial2.0-fitting"
    atomic_json(tmp_path / "manifest.json", manifest)
    _, selected, _ = verify_selected_policy(tmp_path)
    assert selected == chosen


def test_durecdial_text_and_packing_use_shared_native_implementation():
    from tdrl import diffusion_data, durecdial_diffusion_data

    assert durecdial_diffusion_data.pack_dialogues is diffusion_data.pack_dialogues
    assert durecdial_diffusion_data.native_dialogue_text is diffusion_data.native_dialogue_text
