import os
import glob
import cv2
import numpy as np
import mediapipe as mp
import logging
from tqdm import tqdm

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


error_logger = setup_logger("error", "process_errors.log", logging.ERROR)
info_logger = setup_logger("info", "process_info.log", logging.INFO)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def mediapipe_image(frame):
    black = np.zeros(frame.shape)

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=black,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=black,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=black,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style()
            )

        return black


images = glob.glob("DataSet/**/**/**/*.png")


for path in tqdm(images):
    image = cv2.imread(path)

    if image is None:
        error_logger.error(f"Load error for: {path}")
        continue

    processed = mediapipe_image(image)

    if processed is None:
        error_logger.error(f"Process error for: {path}")
        continue

    if not os.path.exists("Processed"):
        os.makedirs("Processed")

    img_cat = path.split(os.path.sep)[1]  # Train, Test or Validation

    if not os.path.exists("Processed"+os.path.sep+img_cat):
        os.makedirs("Processed"+os.path.sep+img_cat)

    save_path = "Processed"+os.path.sep+img_cat + \
        os.path.sep+path.split(os.path.sep)[-1]

    info_logger.info(f"Processed: {path} - {save_path}")
    cv2.imwrite(save_path, processed)
