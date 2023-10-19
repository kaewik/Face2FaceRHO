# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io
import dlib

class DLibFaceDetection:
    def __init__(self, iscrop=True, crop_size=224, scale=1.25, sample_step=10, device='cuda'):
        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.face_boxes = None

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def detect_faces(self, frame):
        self.face_boxes = self.detector(frame, 0)
        first_face_box = self.face_boxes[0]
        bbox = [first_face_box.left(),first_face_box.top(), first_face_box.right(), first_face_box.bottom()]
        return bbox, 'kpt68'

    def get_face_boxes(self):
        return self.face_boxes or []

    def detect(self, frame):
        if self.iscrop:
            bbox, bbox_type = self.detect_faces(frame)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0; right = h-1; top=0; bottom=w-1
            else:
                left = bbox[0]; right=bbox[2]
                top = bbox[1]; bottom=bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])

        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = frame/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # cv2.imwrite("../crop_img.png", dst_image * 255)
        dst_image = dst_image.transpose(2,0,1)
        return {
            'image': torch.tensor(dst_image).float(),
            'tform': torch.tensor(tform.params).float(),
            'original_image': torch.tensor(image.transpose(2,0,1)).float(),
        }
