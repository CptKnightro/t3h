import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

plocX, plocY = 0, 0
clocX, clocY = 0, 0
last_click_time = 0
dragging = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1], lm_list[8][2]
                x2, y2 = lm_list[4][1], lm_list[4][2]
                x3, y3 = lm_list[12][1], lm_list[12][2]
                x4, y4 = lm_list[16][1], lm_list[16][2]
                x0, y0 = lm_list[0][1], lm_list[0][2]
                x5, y5 = lm_list[5][1], lm_list[5][2]

                hand_size = np.hypot(x5 - x0, y5 - y0)
                click_threshold = hand_size * 0.5

                fingers = []
                if lm_list[4][1] > lm_list[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for tip_id in [8, 12, 16, 20]:
                    if lm_list[tip_id][2] < lm_list[tip_id - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                if fingers[1] == 1 and fingers[2] == 1:
                     mid_h = 720 // 2
                     if y1 < mid_h - 50:
                         pyautogui.scroll(20)
                     elif y1 > mid_h + 50:
                         pyautogui.scroll(-20)

                elif fingers[1] == 1:
                    x3_map = np.interp(x1, (0, 1280), (0, screen_w))
                    y3_map = np.interp(y1, (0, 720), (0, screen_h))

                    move_dist = np.hypot(x3_map - plocX, y3_map - plocY)
                    smoothing = 2 if move_dist > 50 else 6

                    clocX = plocX + (x3_map - plocX) / smoothing
                    clocY = plocY + (y3_map - plocY) / smoothing

                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY

                    if np.hypot(x3 - x2, y3 - y2) < click_threshold:
                         if not dragging:
                             pyautogui.mouseDown()
                             dragging = True
                    else:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False

                    if np.hypot(x4 - x2, y4 - y2) < click_threshold:
                         if time.time() - last_click_time > 0.5:
                            pyautogui.rightClick()
                            last_click_time = time.time()

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()