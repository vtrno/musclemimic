import argparse
import logging
import math
import os

import yaml
from omegaconf import DictConfig

from loco_mujoco.smpl.retargeting import extend_motion, fit_gmr_motion_skel, get_skel_model_path
from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.utils import detect_headless_environment, setup_headless_rendering


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SKEL to MyoFullBody retargeting and visualization demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skel-file", type=str, default="demo_json.json", help="Input SKEL motion file")
    parser.add_argument("--output", default="demo_json.npz", help="Output retargeted trajectory")
    parser.add_argument("--record", action="store_true", default=False, help="Record video of trajectory playback")
    parser.add_argument("--output-dir", type=str, default="./retargeting_recordings", help="Output directory for videos")
    parser.add_argument("--video-name", type=str, default="demo_json_skel", help="Name for output video file")
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to play/record")
    parser.add_argument("--n-steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--source-fps", type=float, default=None, help="FPS of the input SKEL motion")
    parser.add_argument("--no-render", action="store_true", default=False, help="Disable rendering after retargeting")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    skel_model_path = get_skel_model_path()
    if not os.path.exists(skel_model_path):
        raise FileNotFoundError(
            f"SKEL models not found at {skel_model_path}. "
            "Set the path with `musclemimic-set-skel-model-path --path /path/to/skel/models`."
        )

    is_headless = detect_headless_environment()
    if is_headless:
        setup_headless_rendering()

    with open("loco_mujoco/smpl/robot_confs/MyoFullBody.yaml") as f:
        robot_conf = DictConfig(yaml.load(f, yaml.SafeLoader))

    skel_config = {
        "src_human": "skel",
        "target_fps": 30,
        "solver": "daqp",
        "damping": 0.5,
        "offset_to_ground": False,
        "use_velocity_limit": False,
        "use_fitted_shape": True,
        "shape_fitting_iterations": 500,
        "root_orientation_offset": None, #[, 0.0, 0.0],
    }
    if args.source_fps is not None:
        skel_config["source_fps"] = args.source_fps

    trajectory, results = fit_gmr_motion_skel(
        'MyoFullBody',
        robot_conf,
        args.skel_file,
        logger,
        skel_config,
    )
    trajectory = extend_motion("MyoFullBody", robot_conf.env_params, trajectory, None)
    trajectory.save(args.output)

    env = MyoFullBody(
        **robot_conf.env_params,
        th_params={"random_start": False, "fixed_start_conf": (0, 0)},
        headless=is_headless,
    )
    env.load_trajectory(traj=trajectory, warn=False)

    print(f"\nTrajectory Info:")
    print(f"  Control frequency: {1.0 / env.dt:.1f} Hz")
    print(f"  Trajectory frequency: {env.th.traj.info.frequency:.1f} Hz")
    print(f"  Trajectory length: {len(env.th.traj.data.qpos)} frames")

    if args.no_render:
        print("\nDry run complete (--no-render specified)")
        return

    recorder_params = None
    if args.record:
        recorder_params = {
            "path": args.output_dir,
            "tag": "myofullbody_skel_retargeted",
            "video_name": args.video_name,
            "compress": True,
        }
        fps = int(1.0 / env.dt)
        print(f"\nRecording to: {args.output_dir}/myofullbody_skel_retargeted/{args.video_name}.mp4 ({fps} FPS)")

    env.play_trajectory(
        n_episodes=args.n_episodes,
        n_steps_per_episode=args.n_steps,
        render=True,
        record=args.record,
        recorder_params=recorder_params,
    )

    if args.record:
        print("Recording completed!")
    else:
        print("Playback completed!")


if __name__ == '__main__':
    main()
