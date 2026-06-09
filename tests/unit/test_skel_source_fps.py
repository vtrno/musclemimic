import loco_mujoco.smpl.retargeting as retargeting


def test_skel_source_fps_overrides_target_fps():
    assert hasattr(retargeting, "_resolve_skel_aligned_fps")
    assert retargeting._resolve_skel_aligned_fps({"source_fps": 100, "target_fps": 30}) == 100.0


def test_skel_source_fps_falls_back_to_target_fps():
    assert hasattr(retargeting, "_resolve_skel_aligned_fps")
    assert retargeting._resolve_skel_aligned_fps({"target_fps": 60}) == 60.0

