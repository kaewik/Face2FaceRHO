import numpy as np
import torchvision.transforms as transforms
import traceback
from options.parse_config import Face2FaceRHOConfigParse
import socket
import pickle
from util.landmark_image_generation import LandmarkImageGeneration
from time import time
import cv2 as cv
import os
from models import create_model
from util.util import tensor2im
import torch

RECV_BUF_SIZE = 1024
RECV_TIMEOUT = 5

class Receiver:
    def __init__(self):
        host = '' # all interfaces
        port = 12345
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))

    def wait_for_sender(self):
        self.s.listen(1)
        self.connection, self.addr = self.s.accept()
        print('Connected by', self.addr)

    def receive_json(self):
        received_data = []
        t_start = time()
        time_ellapsed = 0.0
        while time_ellapsed < RECV_TIMEOUT:
            data = self.connection.recv(RECV_BUF_SIZE)
            received_data.append(data)
            currently_received_message = b''.join(received_data)
            try:
                fully_received_message = pickle.loads(currently_received_message)
                return fully_received_message
            except Exception:
                pass
            time_ellapsed = time() - t_start

        raise Exception("Timeout")

    def close(self):
        if hasattr(self, 'connection'):
            self.connection.close()
        self.s.close()

if __name__ == '__main__':
    config_parse = Face2FaceRHOConfigParse()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    opt = config_parse.get_opt_from_ini(f'{dir_path}/config/test_face2facerho.ini')
    config_parse.setup_environment()

    landmark_img_generator = LandmarkImageGeneration(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    receiver = Receiver()
    try:
        receiver.wait_for_sender()

        key_frame = None
        while True:
            try:
                message = receiver.receive_json()

                if message['type'] == 'key_frame':
                    print('Got key frame')
                    key_frame = pickle.loads(message['data'])
                    cv.imshow('receiver frames', key_frame)
                elif message['type'] == 'face_data':
                    print('Got face data')
                    face_data = pickle.loads(message['data'])
                    src_headpose = torch.from_numpy(np.array(face_data['source']['headpose'])).float()
                    src_landmarks = torch.from_numpy(np.array(face_data['source']['landmarks'])).float()
                    src_landmarks = landmark_img_generator.generate_landmark_img(src_landmarks)
                    src_landmarks = [value.unsqueeze(0) for value in src_landmarks]
                    drv_headpose = torch.from_numpy(np.array(face_data['driving']['headpose'])).float()
                    drv_landmarks = torch.from_numpy(np.array(face_data['driving']['landmarks'])).float()
                    drv_landmarks = landmark_img_generator.generate_landmark_img(drv_landmarks)
                    drv_landmarks = [value.unsqueeze(0) for value in drv_landmarks]
                    img_numpy = np.asarray(key_frame)
                    img_tensor = 2.0 * transforms.ToTensor()(img_numpy.astype(np.float32)) / 255.0 - 1.0
                    model.set_source_face(img_tensor.unsqueeze(0), src_headpose.unsqueeze(0))
                    model.reenactment(src_landmarks, drv_headpose.unsqueeze(0), drv_landmarks)

                    visual_results = model.get_current_visuals()
                    im = tensor2im(visual_results['fake'])
                    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
                    cv.imshow('receiver frames', im)
                else:
                    print(f'Unexpect message received {message}')
                    break
            except socket.error:
                print("Error Occured.")
                break
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                break

            if cv.pollKey() == ord('q'):
                break
    finally:
        receiver.close()
        cv.destroyAllWindows()
