import json
import logging
import os

import numpy as np
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig

from loco_mujoco.smpl.retargeting import extend_motion, fit_gmr_motion_skel

if __name__ == '__main__':
    if 'SKEL_MODEL_PATH' not in os.environ:
        raise FileNotFoundError('Missing `SKEL_MODEL_PATH` in env variables')

    logger = logging.getLogger(__name__)
    with initialize(config_path="fullbody"):
        cfg = compose(config_name="conf_fullbody_skel")

    robot_conf = cfg.experiment
    trajectory, results = fit_gmr_motion_skel(
        'MyoFullBody',
        robot_conf,
        'demo_json.json',
        logger,
        robot_conf.task_factory.params.amass_dataset_conf.gmr_config
    )
    with open("loco_mujoco/smpl/robot_confs/MyoFullBody.yaml") as f:
        robot_conf = DictConfig(yaml.load(f, yaml.SafeLoader))
    trajectory = extend_motion("MyoFullBody", robot_conf.env_params, trajectory, None)
    trajectory.save('demo_json.npz')
