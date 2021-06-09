import cv2
import mediapipe as mp
import time

import numpy as np


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id, bbox, detection.score])

                # img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]


                # cv2.rectangle(img, bbox, (255, 0, 255), 2)
                # cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                #             2, (255, 0, 0), 2)

        return img, bboxs

    # def fancyDraw(self, img, bbox, l=30, t=10):
    #     x, y, w, h = bbox
    #     x1, y1 = x + w, y + h
    #
    #     cv2.rectangle(img, bbox, (255,0,255), 2)
    #     cv2.line(img, (x, y), (x+l,y), (255,0,255), t)
    #
    #     return img



# mpFaceDetection = mp.solutions.face_detection
# mpDraw = mp.solutions.drawing_utils
# faceDetection = mpFaceDetection.FaceDetection()







def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img,bbox = detector.findFaces(img)
        if len(bbox) > 0:
            bbox = bbox[0][1]
            print(bbox)
        # img = img[100:200, 100:200]
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.rectangle(img, (bbox[0],bbox[1], bbox[2],bbox[3]), (255, 0, 255), 2)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    cap.release()
    cv2.DestroyAllWindow()


if __name__ == "__main__":

    # main()
    detector = FaceDetector()
    img = cv2.imread('/home/quan/PycharmProjects/MiAI_FaceRecog_2/video/img.png')
    img = cv2.resize(img, (1024, 800))
    img, bbox = detector.findFaces(img)
    if len(bbox) > 0:
        bbox = bbox[0][1]
        print(bbox)
    cv2.rectangle(img, (bbox[0], bbox[1], bbox[2], bbox[3]), (255, 0, 255), 3)
    crop = img[bbox[1]: bbox[1]+bbox[2], bbox[0]: bbox[0]+bbox[3]]
    img = cv2.resize(img, (1024,800))
    cv2.imshow('img', crop)
    cv2.waitKey(0)