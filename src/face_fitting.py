import torch
import json
import os
import copy
import numpy as np
from external.decalib.utils.config import cfg as deca_cfg
from external.decalib.deca import DECA
from face_detection import DLibFaceDetection
from external.decalib.models.FLAME import FLAME
from util.util import (
    save_coeffs,
    save_landmarks
)

class FLAMEFitting:
    def __init__(self, device="cuda"):
        self.device = device
        self.deca = DECA(config=deca_cfg, device=device)
        self.image_detector = DLibFaceDetection(iscrop=True, sample_step=10, device=self.device)

    def get_face_boxes(self):
        return self.image_detector.get_face_boxes()

    def fitting(self, frame):
        detection_data = self.image_detector.detect(frame)
        images = detection_data['image'].to(self.device)[None, ...]
        with torch.no_grad():
            codedict = self.deca.encode(images)
            codedict['tform'] = detection_data['tform'][None, ...]
            original_image = detection_data['original_image'][None, ...]
            _, _, h, w = original_image.shape
            params = self.deca.ensemble_3DMM_params(codedict, image_size=deca_cfg.dataset.image_size, original_image_size=h)
        return params


class PoseLandmarkExtractor:
    def __init__(self):
        self.flame = FLAME(deca_cfg.model)

        with open(os.path.join(deca_cfg.deca_dir, 'data', 'pose_transform_config.json'), 'r') as f:
            pose_transform = json.load(f)

        self.scale_transform = pose_transform['scale_transform']
        self.tx_transform = pose_transform['tx_transform']
        self.ty_transform = pose_transform['ty_transform']
        self.tx_scale = 0.256 # 512 / 2000
        self.ty_scale = - self.tx_scale

    @staticmethod
    def transform_points(points, scale, tx, ty):
        trans_matrix = torch.zeros((1, 4, 4), dtype=torch.float32)
        trans_matrix[:, 0, 0] = scale
        trans_matrix[:, 1, 1] = -scale
        trans_matrix[:, 2, 2] = 1
        trans_matrix[:, 0, 3] = tx
        trans_matrix[:, 1, 3] = ty
        trans_matrix[:, 3, 3] = 1

        batch_size, n_points, _ = points.shape
        points_homo = torch.cat([points, torch.ones([batch_size, n_points, 1], dtype=points.dtype)], dim=2)
        points_homo = points_homo.transpose(1, 2)
        trans_points = torch.bmm(trans_matrix, points_homo).transpose(1, 2)
        trans_points = trans_points[:, :, 0:3]
        return trans_points

    def get_project_points(self, shape_params, expression_params, pose, scale, tx, ty):
        shape_params = torch.tensor(shape_params).unsqueeze(0)
        expression_params = torch.tensor(expression_params).unsqueeze(0)
        pose = torch.tensor(pose).unsqueeze(0)
        verts, landmarks3d = self.flame(
            shape_params=shape_params, expression_params=expression_params, pose_params=pose)
        trans_landmarks3d = self.transform_points(landmarks3d, scale, tx, ty)
        trans_landmarks3d = trans_landmarks3d.squeeze(0).cpu().numpy()
        return trans_landmarks3d[:, 0:2].tolist()

    def calculate_nose_tip_tx_ty(self, shape_params, expression_params, pose, scale, tx, ty):
        front_pose = copy.deepcopy(pose)
        front_pose[0] = front_pose[1] = front_pose[2] = 0
        front_landmarks3d = self.get_project_points(shape_params, expression_params, front_pose, scale, tx, ty)
        original_landmark3d = self.get_project_points(shape_params, expression_params, pose, scale, tx, ty)
        nose_tx = original_landmark3d[30][0] - front_landmarks3d[30][0]
        nose_ty = original_landmark3d[30][1] - front_landmarks3d[30][1]
        return nose_tx, nose_ty

    def get_pose(self, shape_params, expression_params, pose, scale, tx, ty):
        nose_tx, nose_ty = self.calculate_nose_tip_tx_ty(
            shape_params, expression_params, pose, scale, tx, ty)
        transformed_axis_angle = [
            float(pose[0]),
            float(pose[1]),
            float(pose[2])
        ]
        transformed_tx = self.tx_transform + self.tx_scale * (tx + nose_tx)
        transformed_ty = self.ty_transform + self.ty_scale * (ty + nose_ty)
        transformed_scale = scale / self.scale_transform
        return transformed_axis_angle + [transformed_tx, transformed_ty, transformed_scale]
