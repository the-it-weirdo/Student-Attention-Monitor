import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf


saved_model_path = "./trained_xception.h5"

model = tf.keras.models.load_model(saved_model_path)

class_indices = {0: "['Bored', 'Confused', 'Frustrated']",
                 1: "['Bored', 'Confused']",
                 2: "['Bored', 'Engaged', 'Confused', 'Frustrated']",
                 3: "['Bored', 'Engaged', 'Confused']",
                 4: "['Bored', 'Engaged', 'Frustrated']",
                 5: "['Bored', 'Engaged']",
                 6: "['Bored', 'Frustrated']",
                 7: "['Bored']",
                 8: "['Confused', 'Frustrated']",
                 9: "['Confused']",
                 10: "['Engaged', 'Confused', 'Frustrated']",
                 11: "['Engaged', 'Confused']",
                 12: "['Engaged', 'Frustrated']",
                 13: "['Engaged']",
                 14: "['Frustrated']",
                 15: "['Unknown']"}


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def xception_preprocess(frame):
    return tf.keras.applications.xception.preprocess_input(frame)


def dnn_preprocess():
    pass


def live():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            black = np.zeros((299, 299, 3))
            processed = np.zeros((299, 299, 3))
            predicted_text = "Not Engaged"
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=black,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=black,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=black,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())

                # black.flags.writeable = False
                # preprocess
                processed = xception_preprocess(black.copy())

                if processed is not None:
                    inp = np.array([processed])
                    predicted = model.predict(inp)

                    # predicted_text = np.argmax(predicted)
                    predicted_text = class_indices[np.argmax(predicted)]
                    # print(predicted_text)

                # black.flags.writeable = True
            else:
                predicted_text = "Not Engaged"

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            cv2.putText(image, predicted_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 2)
            cv2.imshow('Camera Capture(Press esc to close)',
                       image)

            cv2.imshow('MediaPipe Face Mesh (Press esc to close)',
                       cv2.flip(black, 1))
            # cv2.imshow('Xception Processed (Press esc to close)',
            #            cv2.flip(processed, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == "__main__":
    print(model.summary())
    live()
