import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

CAM_WIDTH, CAM_HEIGHT = 640, 480
FRAME_REDUCTION = 100  
SMOOTHING = 5          
CLICK_THRESHOLD = 30   
RIGHT_CLICK_THRESHOLD = 30 
SCROLL_SENSITIVITY = 20 

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

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

def get_fingers_up(lm_list):
    """Returns a list [Thumb, Index, Middle, Ring, Pinky] as 0 or 1"""
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
    return fingers

print("System Active: Index=Move | Pinch=L-Click | Middle+Thumb=R-Click | Peace=Scroll")

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    
    cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                  (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION),
                  (255, 0, 255), 2)
    
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
                
                fingers = get_fingers_up(lm_list)
                
                if fingers[1] == 1 and fingers[2] == 0:
                    x3_map = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, screen_w))
                    y3_map = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, screen_h))

                    clocX = plocX + (x3_map - plocX) / SMOOTHING
                    clocY = plocY + (y3_map - plocY) / SMOOTHING
                    
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY
                    
                    dist_click = np.hypot(x2 - x1, y2 - y1)
                    if dist_click < CLICK_THRESHOLD:
                         cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                         if not dragging:
                             pyautogui.mouseDown()
                             dragging = True
                    else:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False

                elif fingers[1] == 1 and fingers[2] == 1:
                    dist_peace = abs(x1 - x3)
                    if dist_peace < 120: 
                         cv2.putText(img, "SCROLL MODE", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                         
                         mid_h = CAM_HEIGHT // 2
                         if y1 < mid_h - 50:
                             pyautogui.scroll(SCROLL_SENSITIVITY)
                         elif y1 > mid_h + 50:
                             pyautogui.scroll(-SCROLL_SENSITIVITY)
                
                dist_right = np.hypot(x2 - x3, y2 - y3)
                if dist_right < RIGHT_CLICK_THRESHOLD:
                    if time.time() - last_click_time > 0.5:
                        cv2.circle(img, (x3, y3), 15, (0, 0, 255), cv2.FILLED)
                        pyautogui.rightClick()
                        last_click_time = time.time()
            
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking v2", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()