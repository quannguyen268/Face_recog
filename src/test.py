from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import pickle
import time
import align.detect_face
from face_detect import FaceDetector
import cv2
import facenet
import imutils
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from imutils.video import VideoStream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = '/home/quan/PycharmProjects/MiAI_FaceRecog_2/Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = '/home/quan/PycharmProjects/MiAI_FaceRecog_2/Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
            detector = FaceDetector()
            people_detected = set()
            person_detected = collections.Counter()

            cap  = VideoStream(src=0).start()
            pTime = 0


            while (True):
                frame = cap.read()




                # frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)
                img, bounding_boxes = detector.findFaces(frame)
                cTime = time.time()
                fps = 1 / (cTime - pTime)

                faces_found = len(bounding_boxes)
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):


                            bb[i][:] = det[i][1][:]


                            cropped = frame[bb[i][1]: bb[i][1]+bb[i][2], bb[i][0]: bb[i][0]+ bb[i][3]]

                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)


                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            print(emb_array.shape)
                            predictions = model.predict_proba(emb_array)

                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            pTime = cTime

                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))


                            if best_class_probabilities > 0.7:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1], bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][1] - 20

                                name = class_names[best_class_indices[0]]


                                cv2.putText(img, f'FPS: {int(fps)}', (bb[i][0], bb[i][1] - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                person_detected[best_name] += 1
                            else:
                                name = "Unknown"

                except:
                    pass

                cv2.imshow('Face Recognition', img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()




main()