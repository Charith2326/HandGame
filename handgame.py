import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks: Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
tip_ids = [4, 8, 12, 16, 20]

# Start webcam
cap = cv2.VideoCapture(0)

def get_finger_status(lm_list):
    fingers = []
    # Thumb
    fingers.append(1 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1] else 0)
    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)
    return fingers

last_action = None
prev_wrist_x = None
move_threshold = 40  # Pixel threshold to detect left/right motion

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    action = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((id, int(lm.x * w), int(lm.y * h)))

            if lm_list:
                fingers = get_finger_status(lm_list)

                # Detect specific finger gestures
                if fingers == [0, 1, 0, 0, 0]:
                    action = "UP"
                    pyautogui.press('up')
                elif fingers == [0, 0, 1, 0, 0]:
                    action = "DOWN"
                    pyautogui.press('down')
                else:
                    # Detect hand movement for LEFT/RIGHT
                    current_wrist_x = lm_list[0][1]  # wrist x position

                    if prev_wrist_x is not None:
                        dx = current_wrist_x - prev_wrist_x

                        if dx > move_threshold:
                            action = "RIGHT"
                            pyautogui.press('right')
                        elif dx < -move_threshold:
                            action = "LEFT"
                            pyautogui.press('left')

                    prev_wrist_x = current_wrist_x

                if action != last_action and action != "None":
                    print(f"Gesture Detected: {action}")
                    last_action = action

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
