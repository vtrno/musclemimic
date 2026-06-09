import yaml

from loco_mujoco.smpl.retargeting import get_skel_model_path


def test_get_skel_model_path_reads_config(monkeypatch, tmp_path):
    config_path = tmp_path / "MUSCLEMIMIC_VARIABLES.yaml"
    skel_path = tmp_path / "skel"
    skel_path.mkdir()
    config_path.write_text(yaml.safe_dump({"MUSCLEMIMIC_SKEL_MODEL_PATH": str(skel_path)}))

    monkeypatch.delenv("SKEL_MODEL_PATH", raising=False)
    monkeypatch.delenv("MUSCLEMIMIC_SKEL_MODEL_PATH", raising=False)
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(config_path))

    assert get_skel_model_path() == str(skel_path)

