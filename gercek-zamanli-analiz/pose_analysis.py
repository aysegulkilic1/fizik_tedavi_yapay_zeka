import cv2
import mediapipe as mp
import math
import numpy as np
import time

STATE_WAITING = 0
STATE_RUNNING = 1
STATE_FINISHED = 2

EXERCISE_NONE = 0
EXERCISE_RIGHT_ARM = 1; EXERCISE_LEFT_ARM = 2
EXERCISE_HEAD_TURN_RIGHT = 3; EXERCISE_HEAD_TURN_LEFT = 4
EXERCISE_RIGHT_LEG = 5; EXERCISE_LEFT_LEG = 6
EXERCISE_HEAD_TILT_RIGHT = 7; EXERCISE_HEAD_TILT_LEFT = 8
EXERCISE_LATERAL_RIGHT = 9; EXERCISE_LATERAL_LEFT = 10
EXERCISE_SIDE_BEND_RIGHT = 11
EXERCISE_SIDE_BEND_LEFT = 12

exercise_sequence = [
    EXERCISE_RIGHT_ARM, EXERCISE_LEFT_ARM,
    EXERCISE_LATERAL_RIGHT, EXERCISE_LATERAL_LEFT,
    EXERCISE_HEAD_TURN_RIGHT, EXERCISE_HEAD_TURN_LEFT,
    EXERCISE_HEAD_TILT_RIGHT, EXERCISE_HEAD_TILT_LEFT,
    EXERCISE_SIDE_BEND_RIGHT, EXERCISE_SIDE_BEND_LEFT,
    EXERCISE_RIGHT_LEG, EXERCISE_LEFT_LEG
]
target_reps_list = [
    10, 10,
    10, 10,
    8, 8,
    8, 8,
    10, 10,
    10, 10
]
set_duration = 120

exercise_names = {
    EXERCISE_NONE: "Beklemede",
    EXERCISE_RIGHT_ARM: "Sag Kol One", EXERCISE_LEFT_ARM: "Sol Kol One",
    EXERCISE_HEAD_TURN_RIGHT: "Kafa Saga Cevir", EXERCISE_HEAD_TURN_LEFT: "Kafa Sola Cevir",
    EXERCISE_RIGHT_LEG: "Sag Bacak", EXERCISE_LEFT_LEG: "Sol Bacak",
    EXERCISE_HEAD_TILT_RIGHT: "Kafa Saga Yatir", EXERCISE_HEAD_TILT_LEFT: "Kafa Sola Yatir",
    EXERCISE_LATERAL_RIGHT: "Sag Kol Yana", EXERCISE_LATERAL_LEFT: "Sol Kol Yana",
    EXERCISE_SIDE_BEND_RIGHT: "Govde Saga Egil", EXERCISE_SIDE_BEND_LEFT: "Govde Sola Egil"
}

mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
RED_COLOR = (0, 0, 255); GREEN_COLOR = (0, 255, 0); BLUE_COLOR = (255, 100, 0)
WHITE_COLOR = (255, 255, 255); BLACK_COLOR = (50, 50, 50); YELLOW_COLOR = (0, 255, 255)
BUTTON_COLOR = (0, 180, 0); BUTTON_TEXT_COLOR = WHITE_COLOR

red_spec_landmark = mp_drawing.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=2)
red_spec_connection = mp_drawing.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=2)
green_spec_landmark = mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=4)
green_spec_connection = mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=2)

def calculate_angle(a, b, c):
    try:
        landmark_a = np.array([a.x, a.y]); landmark_b = np.array([b.x, b.y]); landmark_c = np.array([c.x, c.y])
    except AttributeError:
        landmark_a = np.array(a); landmark_b = np.array(b); landmark_c = np.array(c)
    vector_ba = landmark_a - landmark_b; vector_bc = landmark_c - landmark_b
    radians = math.atan2(vector_bc[1], vector_bc[0]) - math.atan2(vector_ba[1], vector_ba[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_vector_angle(p1, p2, p3, p4):
    try:
        p1_arr = np.array([p1.x, p1.y]); p2_arr = np.array([p2.x, p2.y])
        p3_arr = np.array([p3.x, p3.y]); p4_arr = np.array([p4.x, p4.y])
    except AttributeError:
        p1_arr = np.array(p1); p2_arr = np.array(p2)
        p3_arr = np.array(p3); p4_arr = np.array(p4)
    vector1 = p2_arr - p1_arr; vector2 = p4_arr - p3_arr
    angle1_rad = np.arctan2(vector1[1], vector1[0])
    angle2_rad = np.arctan2(vector2[1], vector2[0])
    angle_rad_diff = angle1_rad - angle2_rad
    while angle_rad_diff > np.pi: angle_rad_diff -= 2 * np.pi
    while angle_rad_diff < -np.pi: angle_rad_diff += 2 * np.pi
    angle_deg = np.degrees(angle_rad_diff)
    return angle_deg

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hata: Kamera açılamadı!")
    exit()

pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

program_state = STATE_WAITING
current_exercise_index = 0
set_start_time = 0
is_set_active = False
current_target_reps = 0

aktif_egzersiz = EXERCISE_NONE
aktif_egzersiz_adi = "Baslatmak icin Butona Tikla"
total_counter = 0

stage_l = "down"; stage_r = "down"; stage_turn = "center"; stage_tilt = "center"; stage_bend = "center"
stage_leg_l = "down"; stage_leg_r = "down"; previous_tilt_stage = "center"; previous_bend_stage = "center"

ELBOW_THRESHOLD_STRAIGHT = 160; SHOULDER_ANGLE_UP = 165; SHOULDER_ANGLE_DOWN = 30
LATERAL_SHOULDER_ANGLE_UP = 170; LATERAL_SHOULDER_ANGLE_DOWN = 30
TURN_METRIC_RIGHT = 0.40; TURN_METRIC_LEFT = 0.40; TURN_METRIC_NEUTRAL = 0.15
KNEE_THRESHOLD_STRAIGHT = 160; HIP_ANGLE_UP = 100; HIP_ANGLE_DOWN = 165
TILT_ANGLE_THRESHOLD = 25; TILT_NEUTRAL_THRESHOLD = 8
BEND_ANGLE_THRESHOLD = 15
BEND_NEUTRAL_THRESHOLD = 5
visibility_threshold = 0.6

def reset_all_stages():
    global stage_l, stage_r, stage_turn, stage_tilt, stage_leg_l, stage_leg_r, previous_tilt_stage, stage_bend, previous_bend_stage
    stage_l = "down"; stage_r = "down"; stage_turn = "center"; stage_tilt = "center"; stage_bend = "center"
    stage_leg_l = "down"; stage_leg_r = "down"; previous_tilt_stage = "center"; previous_bend_stage = "center"

button_w = 150; button_h = 50
start_button_x = None; start_button_y = None; reset_button_x = None; reset_button_y = None

def handle_mouse_click(event, x, y, flags, param):
    global program_state, current_exercise_index, is_set_active, aktif_egzersiz, aktif_egzersiz_adi, total_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        if program_state == STATE_WAITING and start_button_x is not None:
            if start_button_x <= x <= start_button_x + button_w and start_button_y <= y <= start_button_y + button_h:
                print("\n*** Program Başlatılıyor (Buton)! ***")
                program_state = STATE_RUNNING
                current_exercise_index = 0
                is_set_active = False
        elif program_state == STATE_FINISHED and reset_button_x is not None:
            if reset_button_x <= x <= reset_button_x + button_w and reset_button_y <= y <= reset_button_y + button_h:
                print("\n*** Program Yeniden Başlatılıyor (Buton)! ***")
                program_state = STATE_WAITING
                current_exercise_index = 0
                is_set_active = False
                aktif_egzersiz = EXERCISE_NONE
                aktif_egzersiz_adi = "Baslatmak icin Butona Tikla"
                total_counter = 0

window_name = "Fizik Tedavi Takibi - Sirali Setler v4 (Egilme Eklendi, Yuz Bulanik)"

print("MediaPipe Pose başlatıldı - Sirali Egzersiz Modu (Buton Bekleniyor)")
print(f"Toplam {len(exercise_sequence)} egzersiz. Her set maks. {set_duration} sn veya hedef tekrar.")
print("Baslatmak icin EKRANDAKI BUTONA TIKLAYIN, Cikmak icin 'Q' tusuna basin.")

while cap.isOpened():
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("Çıkış yapılıyor (q)...")
        break

    success, image = cap.read()
    if not success:
        print("Kamera karesi alınamadı.")
        continue

    h, w, _ = image.shape
    start_button_x = (w - button_w) // 2
    start_button_y = h // 2 - button_h // 2
    reset_button_x = (w - button_w) // 2
    reset_button_y = h // 2 - button_h // 2 + button_h + 10

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    results = pose.process(image_rgb)
    face_results = face_detection.process(image_rgb)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image.flags.writeable = True

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_roi = image[y:y+h, x:x+w]
            if face_roi.size != 0:
                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                image[y:y+h, x:x+w] = face_roi

    is_pose_correct = False
    remaining_time = 0

    if program_state == STATE_WAITING:
        aktif_egzersiz_adi = "Baslatmak icin Butona Tikla"
        aktif_egzersiz = EXERCISE_NONE
        total_counter = 0
        current_target_reps = 0
        cv2.rectangle(image, (start_button_x, start_button_y), (start_button_x + button_w, start_button_y + button_h), BUTTON_COLOR, -1)
        cv2.putText(image, "BASLAT", (start_button_x + 25, start_button_y + button_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BUTTON_TEXT_COLOR, 2)
    elif program_state == STATE_RUNNING:
        if not is_set_active:
            if current_exercise_index >= len(exercise_sequence):
                program_state = STATE_FINISHED
                aktif_egzersiz_adi = "PROGRAM TAMAMLANDI!"
                aktif_egzersiz = EXERCISE_NONE
                print("\n*** Program Tamamlandı! ***")
                continue
            else:
                aktif_egzersiz = exercise_sequence[current_exercise_index]
                aktif_egzersiz_adi = exercise_names.get(aktif_egzersiz, "?")
                current_target_reps = target_reps_list[current_exercise_index]
                total_counter = 0
                reset_all_stages()
                set_start_time = time.time()
                is_set_active = True
                print(f"\n--- BASLA: {aktif_egzersiz_adi} (Hedef: {current_target_reps}, Sure: {set_duration}s) ---")
        if is_set_active:
            elapsed_time = time.time() - set_start_time
            remaining_time = set_duration - elapsed_time
            if remaining_time <= 0 or total_counter >= current_target_reps:
                is_set_active = False
                current_exercise_index += 1
                result_message = "Sure Doldu" if remaining_time <= 0 else "Hedef Tamam"
                print(f"--- SET BITTI ({aktif_egzersiz_adi}): {result_message}! Yapilan Tekrar: {total_counter} ---")
                aktif_egzersiz = EXERCISE_NONE
                remaining_time = 0
            else:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    try:
                        if aktif_egzersiz == EXERCISE_RIGHT_ARM:
                            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            elbow_r = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                            wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                            hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_r, elbow_r, wrist_r, hip_r]):
                                elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                                is_elbow_straight_r = elbow_angle_r > ELBOW_THRESHOLD_STRAIGHT
                                if is_elbow_straight_r:
                                    if shoulder_angle_r > SHOULDER_ANGLE_UP:
                                        is_pose_correct = True
                                        stage_r = "up" if stage_r == 'down' else stage_r
                                    elif shoulder_angle_r < SHOULDER_ANGLE_DOWN and stage_r == 'up':
                                        stage_r = "down"
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_LEFT_ARM:
                            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                            wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                            hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_l, elbow_l, wrist_l, hip_l]):
                                elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                                shoulder_angle_l = calculate_angle(hip_l, shoulder_l, elbow_l)
                                is_elbow_straight_l = elbow_angle_l > ELBOW_THRESHOLD_STRAIGHT
                                if is_elbow_straight_l:
                                    if shoulder_angle_l > SHOULDER_ANGLE_UP:
                                        is_pose_correct = True
                                        stage_l = "up" if stage_l == 'down' else stage_l
                                    elif shoulder_angle_l < SHOULDER_ANGLE_DOWN and stage_l == 'up':
                                        stage_l = "down"
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_HEAD_TURN_RIGHT or aktif_egzersiz == EXERCISE_HEAD_TURN_LEFT:
                            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            if all(lm.visibility > visibility_threshold for lm in [nose, left_ear, right_ear, shoulder_l, shoulder_r]):
                                shoulder_width = abs(shoulder_l.x - shoulder_r.x)
                                if shoulder_width > 0.05:
                                    dist_nose_r_ear = nose.x - right_ear.x
                                    dist_l_ear_nose = left_ear.x - nose.x
                                    turn_metric = (dist_nose_r_ear - dist_l_ear_nose) / shoulder_width
                                    if turn_metric > TURN_METRIC_RIGHT and stage_turn == 'center':
                                        stage_turn = 'right'
                                    elif turn_metric < -TURN_METRIC_LEFT and stage_turn == 'center':
                                        stage_turn = 'left'
                                    returned_to_center = abs(turn_metric) < TURN_METRIC_NEUTRAL
                                    if aktif_egzersiz == EXERCISE_HEAD_TURN_RIGHT and turn_metric > TURN_METRIC_RIGHT:
                                        is_pose_correct = True
                                    elif aktif_egzersiz == EXERCISE_HEAD_TURN_LEFT and turn_metric < -TURN_METRIC_LEFT:
                                        is_pose_correct = True
                                    if returned_to_center:
                                        if aktif_egzersiz == EXERCISE_HEAD_TURN_RIGHT and stage_turn == 'right':
                                            stage_turn = 'center'
                                            total_counter += 1
                                        elif aktif_egzersiz == EXERCISE_HEAD_TURN_LEFT and stage_turn == 'left':
                                            stage_turn = 'center'
                                            total_counter += 1
                                        elif stage_turn != 'center':
                                            stage_turn = 'center'
                        elif aktif_egzersiz == EXERCISE_RIGHT_LEG:
                            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                            knee_r = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                            ankle_r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_r, hip_r, knee_r, ankle_r]):
                                knee_angle_r = calculate_angle(hip_r, knee_r, ankle_r)
                                hip_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)
                                is_knee_straight_r = knee_angle_r > KNEE_THRESHOLD_STRAIGHT
                                if is_knee_straight_r:
                                    if hip_angle_r < HIP_ANGLE_UP and stage_leg_r == 'down':
                                        stage_leg_r = "up"
                                        is_pose_correct = True
                                    elif hip_angle_r > HIP_ANGLE_DOWN and stage_leg_r == 'up':
                                        stage_leg_r = "down"
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_LEFT_LEG:
                            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                            knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                            ankle_l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_l, hip_l, knee_l, ankle_l]):
                                knee_angle_l = calculate_angle(hip_l, knee_l, ankle_l)
                                hip_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)
                                is_knee_straight_l = knee_angle_l > KNEE_THRESHOLD_STRAIGHT
                                if is_knee_straight_l:
                                    if hip_angle_l < HIP_ANGLE_UP and stage_leg_l == 'down':
                                        stage_leg_l = "up"
                                        is_pose_correct = True
                                    elif hip_angle_l > HIP_ANGLE_DOWN and stage_leg_l == 'up':
                                        stage_leg_l = "down"
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_HEAD_TILT_RIGHT or aktif_egzersiz == EXERCISE_HEAD_TILT_LEFT:
                            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                            if all(lm.visibility > visibility_threshold for lm in [left_ear, right_ear, shoulder_l, shoulder_r]):
                                tilt_angle = calculate_vector_angle(left_ear, right_ear, shoulder_l, shoulder_r)
                                current_stage_tilt = stage_tilt
                                if tilt_angle > TILT_ANGLE_THRESHOLD and stage_tilt == "center":
                                    stage_tilt = "right"
                                elif tilt_angle < -TILT_ANGLE_THRESHOLD and stage_tilt == "center":
                                    stage_tilt = "left"
                                elif abs(tilt_angle) < TILT_NEUTRAL_THRESHOLD and stage_tilt != "center":
                                    previous_tilt_stage = stage_tilt
                                    stage_tilt = "center"
                                    returned_to_center = True
                                else:
                                    returned_to_center = False
                                if aktif_egzersiz == EXERCISE_HEAD_TILT_RIGHT and stage_tilt == "right":
                                    is_pose_correct = True
                                elif aktif_egzersiz == EXERCISE_HEAD_TILT_LEFT and stage_tilt == "left":
                                    is_pose_correct = True
                                if returned_to_center:
                                    if aktif_egzersiz == EXERCISE_HEAD_TILT_RIGHT and previous_tilt_stage == "right":
                                        total_counter += 1
                                    elif aktif_egzersiz == EXERCISE_HEAD_TILT_LEFT and previous_tilt_stage == "left":
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_LATERAL_RIGHT:
                            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            elbow_r = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                            wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                            hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_r, elbow_r, wrist_r, hip_r]):
                                elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                                is_elbow_straight_r = elbow_angle_r > ELBOW_THRESHOLD_STRAIGHT
                                if is_elbow_straight_r:
                                    if shoulder_angle_r > LATERAL_SHOULDER_ANGLE_UP and stage_r == 'down':
                                        is_pose_correct = True
                                        stage_r = "up"
                                    elif shoulder_angle_r < LATERAL_SHOULDER_ANGLE_DOWN and stage_r == 'up':
                                        stage_r = "down"
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_LATERAL_LEFT:
                            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                            wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                            hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_l, elbow_l, wrist_l, hip_l]):
                                elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                                shoulder_angle_l = calculate_angle(hip_l, shoulder_l, elbow_l)
                                is_elbow_straight_l = elbow_angle_l > ELBOW_THRESHOLD_STRAIGHT
                                if is_elbow_straight_l:
                                    if shoulder_angle_l > LATERAL_SHOULDER_ANGLE_UP and stage_l == 'down':
                                        is_pose_correct = True
                                        stage_l = "up"
                                    elif shoulder_angle_l < LATERAL_SHOULDER_ANGLE_DOWN and stage_l == 'up':
                                        stage_l = "down"
                                        total_counter += 1
                        elif aktif_egzersiz == EXERCISE_SIDE_BEND_RIGHT or aktif_egzersiz == EXERCISE_SIDE_BEND_LEFT:
                            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                            hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                            if all(lm.visibility > visibility_threshold for lm in [shoulder_l, shoulder_r, hip_l, hip_r]):
                                mid_shoulder_x = (shoulder_l.x + shoulder_r.x) / 2
                                mid_shoulder_y = (shoulder_l.y + shoulder_r.y) / 2
                                mid_hip_x = (hip_l.x + hip_r.x) / 2
                                mid_hip_y = (hip_l.y + hip_r.y) / 2
                                spine_vector_x = mid_shoulder_x - mid_hip_x
                                spine_vector_y = mid_shoulder_y - mid_hip_y
                                bend_angle = np.degrees(np.arctan2(spine_vector_y, spine_vector_x))
                                UPRIGHT_ANGLE_CENTER = -90
                                UPRIGHT_LOWER = UPRIGHT_ANGLE_CENTER - BEND_NEUTRAL_THRESHOLD
                                UPRIGHT_UPPER = UPRIGHT_ANGLE_CENTER + BEND_NEUTRAL_THRESHOLD
                                BEND_RIGHT_THRESH = UPRIGHT_ANGLE_CENTER + BEND_ANGLE_THRESHOLD
                                BEND_LEFT_THRESH = UPRIGHT_ANGLE_CENTER - BEND_ANGLE_THRESHOLD
                                current_stage_bend = stage_bend
                                returned_to_center = False
                                if bend_angle > BEND_RIGHT_THRESH and stage_bend == "center":
                                    stage_bend = "right"
                                elif bend_angle < BEND_LEFT_THRESH and stage_bend == "center":
                                    stage_bend = "left"
                                elif UPRIGHT_LOWER < bend_angle < UPRIGHT_UPPER and stage_bend != "center":
                                    previous_bend_stage = stage_bend
                                    stage_bend = "center"
                                    returned_to_center = True
                                if aktif_egzersiz == EXERCISE_SIDE_BEND_RIGHT and stage_bend == "right":
                                    is_pose_correct = True
                                elif aktif_egzersiz == EXERCISE_SIDE_BEND_LEFT and stage_bend == "left":
                                    is_pose_correct = True
                                if returned_to_center:
                                    if aktif_egzersiz == EXERCISE_SIDE_BEND_RIGHT and previous_bend_stage == "right":
                                        total_counter += 1
                                        print(f"SIDE BEND R OK: {total_counter}")
                                    elif aktif_egzersiz == EXERCISE_SIDE_BEND_LEFT and previous_bend_stage == "left":
                                        total_counter += 1
                                        print(f"SIDE BEND L OK: {total_counter}")

                    except Exception as e:
                        pass

                    landmark_spec = green_spec_landmark if is_pose_correct else red_spec_landmark
                    connection_spec = green_spec_connection if is_pose_correct else red_spec_connection
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              landmark_drawing_spec=landmark_spec, connection_drawing_spec=connection_spec)

    elif program_state == STATE_FINISHED:
        aktif_egzersiz_adi = "PROGRAM TAMAMLANDI!"
        aktif_egzersiz = EXERCISE_NONE
        total_counter = 0
        current_target_reps = 0
        remaining_time = 0
        cv2.rectangle(image, (reset_button_x, reset_button_y), (reset_button_x + button_w, reset_button_y + button_h), BUTTON_COLOR, -1)
        cv2.putText(image, "TEKRAR", (reset_button_x + 20, reset_button_y + button_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BUTTON_TEXT_COLOR, 2)

    box_h = 110; box_w = 350
    cv2.rectangle(image, (0, 0), (box_w, box_h), BLACK_COLOR, -1)
    if program_state == STATE_WAITING:
        status_text = "Baslatmak icin BUTONA Tikla"
        status_color = YELLOW_COLOR
    elif program_state == STATE_FINISHED:
        status_text = "PROGRAM TAMAMLANDI!"
        status_color = YELLOW_COLOR
    else:
        status_text = f"Egzersiz: {aktif_egzersiz_adi}"
    status_color = BLUE_COLOR
    cv2.putText(image, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    if program_state == STATE_RUNNING and is_set_active:
        mins = int(remaining_time // 60)
        secs = int(remaining_time % 60)
        time_text = f"Kalan Sure: {mins:02d}:{secs:02d}"
    else:
        time_text = "Kalan Sure: --:--"
    cv2.putText(image, time_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 2)
    if program_state == STATE_RUNNING:
        rep_text = f"Tekrar: {total_counter} / {current_target_reps}"
        rep_color = GREEN_COLOR if is_set_active and total_counter >= current_target_reps else WHITE_COLOR
        if not is_set_active and current_exercise_index > 0:
            prev_target = target_reps_list[current_exercise_index-1] if current_exercise_index > 0 else 0
            rep_text = f"Tekrar: {total_counter} / {prev_target} (Bitti)"
            rep_color = GREEN_COLOR
    else:
        rep_text = "Tekrar: -- / --"
        rep_color = WHITE_COLOR
    cv2.putText(image, rep_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rep_color, 2)
    if program_state == STATE_RUNNING and is_set_active:
        tick_text = "+" if is_pose_correct else "-"
        tick_color = GREEN_COLOR if is_pose_correct else RED_COLOR
        cv2.putText(image, tick_text, (box_w - 70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tick_color, 3)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, handle_mouse_click)

cap.release()
cv2.destroyAllWindows()
pose.close()
face_detection.close()
print("Uygulama kapatıldı.")
