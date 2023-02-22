import tkinter as tk
import customtkinter as ck
import os, time, cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
import tensorflow as tf
from threading import Thread 


'''
TODO: CÀI ĐẶT THƯ VIỆN NUMPY 
'''
#Cách 1:
!pip install numpy

#Cách 2:
!conda install numpy 

#======><======

'''
TODO: ĐỂ SỬ DỤNG NUMPY TRONG PYTHON
'''

import numpy as np


def reset_sequence():
    global sequence_frames
    sequence_frames = []

def get_reply():
    reply_answer.configure(text = "> " + entry.get())

def on_detect(): 
    global tmp_detect
    tmp_detect = 1 

def clear_all():
    reset_sequence()
    global output_detect
    global reply_answer
    global labels
    labels = []
    output_detect.configure(text="   ...                                    ")
    reply_answer.configure(text = "  ... ")

   
image_path = "D:\Personalproject\mlapp-main/GUI/LOGO_HCMUTE.png"


#=========== Desgin GUI ==========

ck.set_appearance_mode("system")  # Modes: system (default), light, dark
ck.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

root = ck.CTk()  # create CTk window like you do with the Tk window
root.geometry("1280x850")
root.title("Đồ án tốt nghiệp HCMUTE")


def change_appearance_mode_event(new_appearance_mode: str):
        ck.set_appearance_mode(new_appearance_mode)

#========== HEAD FRAME ==========
head_frame = ck.CTkFrame(master = root, width=200, height=200, corner_radius=10)
head_frame.grid(row =0, column=0, rowspan=3, columnspan=20, sticky="nwne", padx=20, pady=20)

title_1 = ck.CTkLabel(head_frame, text="Đề tài:", compound="right", font=ck.CTkFont(size=20, weight="bold"))
title_1.grid(row=0, column=0, padx=10, pady=10)

name_title = ck.CTkLabel(head_frame, text="NGHIÊN CỨU, PHÁT TRIỂN HỆ THỐNG HỖ TRỢ NGƯỜI KHIẾM THÍNH ỨNG DỤNG TRÍ TUỆ NHÂN TẠO", compound="right", font=ck.CTkFont(size=25, weight="bold"))
name_title.grid(row=1, column=1, columnspan = 7 , padx=20, pady=10)


#========== EXTRACT KEYPOINTS FRAME ==========
extract_frame = ck.CTkFrame(master = root, width=600, height= 600, corner_radius=10)
extract_frame.grid(row=3, column=0, rowspan=10, columnspan = 2, sticky="nsew", padx=10, pady=10)
extract_frame.grid_rowconfigure(4, weight=1)

lmain = tk.Label(extract_frame) 
lmain.grid(row=4, column=0, rowspan=5,columnspan=2, padx=10, pady=10)

sidebar_button_label = ck.CTkLabel(extract_frame, text="Detect Hand Sign:", font=ck.CTkFont(size=20, weight="bold"), anchor="w")
sidebar_button_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")

sidebar_button_1 = ck.CTkButton(extract_frame,text="Detect" , command=on_detect)
sidebar_button_1.grid(row=9, column=1, padx=20, pady=10, sticky="e")



#========== DISPLAY OUTPUT FRAMES ==========
display_frame = ck.CTkFrame(master = root, width=600, height= 600, corner_radius=10)
display_frame.grid(row=3, column=2, rowspan=5, columnspan = 2, sticky="nsew", padx=10, pady=10)

display_frame_label = ck.CTkLabel(display_frame, text="Output:", font=ck.CTkFont(size=15, weight="bold"))
display_frame_label.grid(row=3, column=2, padx=20, pady=(20, 10), sticky="nw")

output_detect = ck.CTkLabel(display_frame, text= "Xin chào - Tôi là người khiếm thính! - ...                               ", font=ck.CTkFont(size=25, weight="bold"))
output_detect.grid(row=4, column=2, columnspan=4, padx=20, pady=(20, 10), sticky="nw")


#========== REPLY FRAMES ==========
reply_frame = ck.CTkFrame(master = root, width=600, height= 600, corner_radius=10)
reply_frame.grid(row=8, column=2, rowspan=5, columnspan = 6, sticky="nsew", padx=10, pady=10)
# reply_frame.grid_rowconfigure(4, weight=1)

reply_frame_label = ck.CTkLabel(reply_frame, text="Reply:", font=ck.CTkFont(size=20, weight="bold"))
reply_frame_label.grid(row=8, column=2, padx=20, pady=(20, 10), sticky="nw")

entry = ck.CTkEntry(master=reply_frame,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)

entry = ck.CTkEntry(reply_frame, placeholder_text="Type the answer here")
entry.grid(row=9, column=2, columnspan=5, padx=(20, 20), pady=(20, 20), sticky="ew")
main_button_1 = ck.CTkButton(master=reply_frame,text= "Answer",fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=get_reply)
main_button_1.grid(row=9, column=8, padx=(5, 5), pady=(20, 20), sticky="")

reply_answer = ck.CTkLabel(reply_frame, text="...", font=ck.CTkFont(size=30, weight="bold"))
reply_answer.grid(row=11, column=2, columnspan=2, padx=20, pady=(20, 10), sticky="nw")


#========== SIDEBAR FRAMES ==========

sidebar_frame = ck.CTkFrame(master = root, width=350, height= 350, corner_radius=10)
sidebar_frame.grid(row=3, column=8, rowspan=8, columnspan = 10, sticky="nsew", padx=10, pady=10)

logo_frame = ck.CTkFrame(master = sidebar_frame, width=100, height= 100, corner_radius=10)
logo_frame.grid(row=3, column=9, columnspan=2, padx=20, pady=10, sticky="n")

large_test_image = ck.CTkImage(Image.open(image_path), size=(100, 100))
home_frame_large_image_label = ck.CTkLabel(logo_frame, text="", image=large_test_image)
home_frame_large_image_label.pack()

appearance_mode_label = ck.CTkLabel(sidebar_frame, text="Appearance Mode:", anchor="w")
appearance_mode_label.grid(row=4, column=9, padx=20, pady=(10, 10), sticky="w")

radio_button_1 = ck.CTkOptionMenu(master=sidebar_frame, values=["Light", "Dark", "System"],command=change_appearance_mode_event)
radio_button_1.grid(row=4, column=10, pady=20, padx=20, sticky="w")

sidebar_button_1 = ck.CTkButton(sidebar_frame,text="RESET / CLEAR ALL" , command=clear_all)
sidebar_button_1.grid(row=5, column=10, padx=20, pady=10, sticky="e")

reset_label = ck.CTkLabel(sidebar_frame, text="Reset / Clear all:", anchor="w")
reset_label.grid(row=5, column=9, padx=20, pady=(10, 10), sticky="w")

shutdown_label = ck.CTkLabel(sidebar_frame, text="Shut Down:", anchor="w")
shutdown_label.grid(row=6, column=9, padx=20, pady=(10, 10), sticky="w")

sidebar_button_2 = ck.CTkButton(sidebar_frame,text="SHUT DOWN" , command=root.destroy)
sidebar_button_2.grid(row=6, column=10, padx=20, pady=10, sticky="e")


#========== Setup and function to extract keypoints ==========
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5) 

model = tf.keras.models.load_model("action.h5")
video_path = "D:/Personalproject/MS-ASL/Videos/video_split/deaf/00019.mp4"
cap = cv2.VideoCapture(video_path)
sequence_frames = []
tmp_detect = 0
str_in_output = ""
labels = []

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21



def reset_tmp_detect():
    global tmp_detect
    tmp_detect = 0

def process_body_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.array([p.visibility for p in landmarks])
    return kps, conf


def process_other_landmarks(component, n_points):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.ones(n_points)
    return kps, conf

def get_landmark(results):
    global N_FACE_LANDMARKS, N_BODY_LANDMARKS, N_HAND_LANDMARKS

    keypoints = []
    confs = []

    body_data, body_conf = process_body_landmarks(
        results.pose_landmarks, N_BODY_LANDMARKS
    )
    lh_data, lh_conf = process_other_landmarks(
        results.left_hand_landmarks, N_HAND_LANDMARKS
    )
    rh_data, rh_conf = process_other_landmarks(
            results.right_hand_landmarks, N_HAND_LANDMARKS
        )

    data = np.concatenate([body_data, lh_data, rh_data])
    conf = np.concatenate([body_conf, lh_conf, rh_conf])

    keypoints.append(data)
    confs.append(conf)
    return keypoints

def mediapipe_def(frame):
    global sequence_frames

    results = holistic.process(frame)

    keypoints = get_landmark(results)
    sequence_frames.append(keypoints)
   

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5), 
        mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10)) 

    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

    cv2.putText(frame,'Detecting!', (35,23), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,255),1,2)

    cv2.circle(frame, (20, 20), 9, (255, 0, 0), -1)

    cv2.putText(frame,'Frame: {}'.format(len(sequence_frames)), (360,23), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255),1,2)
    
    img = frame[:, :460, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    # lmain.after(10, detect)
    return keypoints

def make_landmark_timestep(results, frame):
    global sequence_frames
    c_lm=[]
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5), 
        mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10)) 

    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

    cv2.putText(frame,'Detecting!', (35,23), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 0, 0),1,2)

    cv2.circle(frame, (20, 20), 9, (255, 0, 0), -1)

    # cv2.putText(frame,'Frame: {}'.format(len(sequence_frames)), (360,23), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255),1,2)
    
    img = frame[:, :460, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)

    return c_lm

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

actions = ["Xin chao", "Tôi khiếm thính", "Tôi khỏe"]

def detect_sign(model, frames):
    global labels
    global str_in_output
    global output_detect
    global actions
    global sequence_frames

    res = model.predict(np.expand_dims(frames, axis=0))[0]
    label = actions[np.argmax(res)]
    
    labels.append(label)
    str_in_output = " - ".join(labels)
    output_detect.configure(text=str_in_output)# +" ....          ")
    # sequence_frames = 0

    return labels, str_in_output


def detect_2(): 
    global current_stage
    global tmp_detect
    global sequence_frames
    global labels, str_in_output
    global output_detect
    # sequence_frame
    ret, image = cap.read()
    image = cv2.resize(image, (480, 520))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    if tmp_detect ==1:
        
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        sequence_frames.append(keypoints)
        c_lm = make_landmark_timestep(results, image)
        
        if len(sequence_frames) == 30:

            # predict
            try:
                # sequence_frames = sequence_frames[:10]
                t1 = Thread(target=detect_sign, args=(model, sequence_frames,))
                t1.start()

                print("done!")
                print("===============")
                print("\n")

                reset_tmp_detect()
                reset_sequence()
            except Exception as e: 
                print(e) 

    if tmp_detect == 0:
        img = image[:, :460, :] 
        imgarr = Image.fromarray(img) 
        imgtk = ImageTk.PhotoImage(imgarr) 
        lmain.imgtk = imgtk 
        lmain.configure(image=imgtk)
    

    root.after(10, detect_2)    

detect_2()

root.mainloop()