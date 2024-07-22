# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:30:56 2024

@author: leehj
"""

# 객체 추적 예제 (c3_objectTracking.py)

# 관련 라이브러리 선언
import cv2
from createFolder import createFolder

save_dir = "../code_res_imgs/c3_objectTracking"
createFolder(save_dir)

# 추적 방법 나열
#tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

# 'CSRT', 'KCF'
tracker_types = ['CSRT', 'KCF']

for tracker_type in tracker_types:
    # 추적기 생성
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # 비디오 읽기
    cap = cv2.VideoCapture("../images/video8.mp4")

    # 첫프레임을 읽고, 추적 영역 설정
    ret, frame = cap.read()
    if not ret:
        exit(-1)
    frame = cv2.resize(frame, (320, 240))
    bbox = cv2.selectROI(frame, False)
    while bbox == (0,0,0,0):
        bbox = cv2.selectROI(frame, False)

    # 추적기 초기화
    ok = tracker.init(frame, bbox)

    # 비디오 프레임을 불러와서 추적 실행
    indexImg = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break

        # 처리시간 측정
        timer = cv2.getTickCount()

        # 추적 실행
        frame = cv2.resize(frame, (320, 240))
        ret_tracker, bbox = tracker.update(frame)

        # 처리시간 측정
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 추적 결과 표시
        if ret_tracker:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 추적 정보 표시
        cv2.putText(frame, tracker_type + " Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Tracking", frame)
        cv2.imwrite(save_dir + "/" + str(tracker_type) + "_" + str(indexImg) + ".jpg", frame)
        indexImg += 1

        # ESC 버튼 입력시 종료
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break