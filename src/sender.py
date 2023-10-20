from face_fitting import FLAMEFitting, PoseLandmarkExtractor
from imutils import face_utils
import imutils
import time
import dlib
import cv2 as cv
import socket
import pickle

MAX_HEIGHT = 600
MAX_WIDTH = 600

class Sender:
    def __init__(self):
        self.host = socket.gethostname()
        self.port = 12345
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.s.connect((self.host, self.port))

    def send_source_frame(self, frame):
        message = {
            'type': 'key_frame',
            'data': pickle.dumps(frame)
        }
        self.s.sendall(pickle.dumps(message))

    def send_face_data(self, face_data):
        message = {
            'type': 'face_data',
            'data': pickle.dumps(face_data)
        }
        self.s.sendall(pickle.dumps(message))

    def close(self):
        self.s.close()


def calculate_face_data(face_fitting, pose_lml_extractor, src_img, drv_img):
    # 3DMM fitting by DECA: Detailed Expression Capture and Animation using FLAME model
    src_params = face_fitting.fitting(src_img)
    drv_params = face_fitting.fitting(drv_img)

    # calculate head pose and facial landmarks for the source and driving face images
    src_headpose = pose_lml_extractor.get_pose(
        src_params['shape'], src_params['exp'], src_params['pose'],
        src_params['scale'], src_params['tx'], src_params['ty'])

    src_lmks = pose_lml_extractor.get_project_points(
        src_params['shape'], src_params['exp'], src_params['pose'],
        src_params['scale'], src_params['tx'], src_params['ty'])

    # Note that the driving head pose and facial landmarks are calculated using the shape parameters of the source image
    # in order to eliminate the interference of the driving actor's identity.
    drv_headpose = pose_lml_extractor.get_pose(
        src_params['shape'], drv_params['exp'], drv_params['pose'],
        drv_params['scale'], drv_params['tx'], drv_params['ty'])

    drv_lmks = pose_lml_extractor.get_project_points(
        src_params['shape'], drv_params['exp'], drv_params['pose'],
        drv_params['scale'], drv_params['tx'], drv_params['ty'])

    return {
        "source": {
            "headpose": src_headpose,
            "landmarks": src_lmks,
        },
        "driving": {
            "headpose": drv_headpose,
            "landmarks": drv_lmks,
        },
    }

if __name__ == '__main__':
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    #print("[INFO] loading facial landmark predictor...")
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_fitting = FLAMEFitting(device="cpu")
    pose_lml_extractor = PoseLandmarkExtractor()
    cap = cv.VideoCapture(0)
    # Limit Capture FPS to 25
    cap.set(cv.CAP_PROP_FPS, 25)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)
    time.sleep(2.0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    sender = Sender()
    try:
        sender.connect()
        source_frame = None
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
               print("Can't receive frame (stream end?). Exiting ...")
               break
            #small_frame = imutils.resize(frame, width=WIDTH)
            h, w, _ = frame.shape
            size = min(h,w)
            height_offset = h - size
            height_margin = int(height_offset / 2)
            width_offset = w - size
            width_margin = int(width_offset / 2)
            small_frame = frame[height_margin:h - height_margin, width_margin: w - width_margin]
            if source_frame is None:
                source_frame = small_frame
                sender.send_source_frame(source_frame)
            else:
                gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
                gray_source_frame = cv.cvtColor(source_frame, cv.COLOR_BGR2GRAY)
                #face_data = calculate_face_data(face_fitting, pose_lml_extractor, gray_source_frame, gray_frame)

                # detect faces in the grayscale frame
                #face_boxes = detector(gray_frame, 0)
                # loop over the face detections
                #for face_box in face_boxes:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    #shape = predictor(gray_frame, face_box)
                    #shape = face_utils.shape_to_np(shape)
                face_data = calculate_face_data(face_fitting, pose_lml_extractor, source_frame, small_frame)
                    # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw them on the image
                #cv.rectangle(small_frame, (face_box.left(), face_box.top()), (face_box.right(), face_box.bottom()), color=(0, 0, 255), thickness=1)
                #for face_box in face_fitting.get_face_boxes():
                #    face_box_width = face_box.right() - face_box.left()
                #    x_center = int(face_box.right() - face_box_width / 2)
                #    face_box_height = face_box.top() - face_box.bottom()
                #    y_center = int(face_box.top() - face_box_height / 2)
                #    center_point = (x_center, y_center)
                #    for (x, y) in shape["driving"]["landmarks"]:
                #        scale = 2000 / 512
                #        x_offset_from_center = int(x * scale * face_box_width / 2.0)
                #        y_offset_from_center = int(y * scale * face_box_height / 2.0)
                #        x_coord = x_center + x_offset_from_center
                #        y_coord = y_center - y_offset_from_center
                #        cv.circle(small_frame, (x_coord, y_coord), 1, (0, 0, 255), -1)
                sender.send_face_data(face_data)

            # Display the resulting frame
            cv.imshow('sender frames', small_frame)

            if cv.pollKey() == ord('q'):
                break
    finally:
        sender.close()
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()
