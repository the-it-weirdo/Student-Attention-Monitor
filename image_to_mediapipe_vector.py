import pandas as pd
import numpy as np
import mediapipe as mp
import logging
import cv2
from tqdm import tqdm
import os


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


error_logger = setup_logger("error", "raw_process_errors.log", logging.ERROR)
info_logger = setup_logger("info", "raw_process_info.log", logging.INFO)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def mediapipe_image(frame, filepath):
    abs_path = os.path.abspath(filepath)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            error_logger.error("Skipped "+abs_path +
                               '/. Error: ' + "Result None")
            return None
        # landmark vector with xyz
        return results.multi_face_landmarks[0].landmark
        # for face_landmarks in results.multi_face_landmarks:

        # print(results)


tqdm.pandas()
chunksize = 1000

with pd.read_csv("Raw_Multilabel20.csv", chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        chunk["Mediapipe Output"] = chunk["Filepath"].progress_apply(
            lambda x: mediapipe_image(cv2.imread(x), x))

        chunk.to_csv(
            f"Mediapipe_vector/Medipipe_vector_out_chunk_{i}.csv", index=False)
