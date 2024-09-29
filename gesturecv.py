# gesture_math.py

import cv2
import cvzone
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
import streamlit as st

# Configure Generative AI API Key
genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return cap

def get_hand_info(detector, img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw_hand_gesture(info, prev_pos, canvas, img):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
    return current_pos, canvas

def send_to_ai(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

def run_gesture_math_solver():
    # st.set_page_config(layout="wide")
    st.image('MathGestures.png')

    col1, col2 = st.columns([3, 2])
    with col1:
        run = st.checkbox('Run', value=True)
        FRAME_WINDOW = st.image([])

    with col2:
        st.title("Answer")
        output_text_area = st.subheader("")

    cap = initialize_camera()
    prev_pos = None
    canvas = None
    output_text = ""

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        info = get_hand_info(detector, img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw_hand_gesture(info, prev_pos, canvas, img)
            output_text = send_to_ai(model, canvas, fingers)

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

        if output_text:
            output_text_area.text(output_text)

        cv2.waitKey(1)
