#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
Dependencies needed to run the script
'''
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np


# In[3]:


'''
check if gpu is available helps alot
with the frame rate once we use the webcam
if you dont have it just install the needed cuda and cuDNN
'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    device = torch.device("cuda:0")  # Set device to GPU 1
print("Torch is using device:", torch.cuda.get_device_name(device))


# In[7]:


'''
running the model through cuda AKA the gpu for the frame rate fix mentioned above
'''
model = YOLO(r"bank_notes_v2.pt")
model = model.to("cuda")


# In[9]:


'''
previously used to detect banknotes but not as good as tracker 
'''
# model.predict(source, save=True, imgsz=320, conf=0.5)


# In[15]:


import cv2
import numpy as np
import time

'''
initializing the cam and showing a sepearte screen to display the sum value
next, map each class output to the banknote denomination 
'''
cap = cv2.VideoCapture(0)

cv2.namedWindow('Value Display', cv2.WINDOW_NORMAL)

# Define the value mappings
value_mappings = {
    0: 1,
    1: 5,
    2: 10,
    3: 20,
    4: 50
}



cumulative_sum = 0
last_predicted_class = None
last_detection_time = time.time()
max_frames_without_detection = 30  # Maximum number of frames without detection
reset_timeout = 5  # Time in seconds to reset the cumulative sum

while True:
    
    ret, frame = cap.read()
    '''
    using the built in .track from yolo to predict classes and modify the confidence level for the prediction
    then iterating through each result and getting its bbox which includes the class
    '''
    results = model.track(source=frame, conf=0.7, iou=0.5, show=False)
    class_pred = None  # Initialize class_pred outside the loop
    for result in results:
        boxes = result.boxes.data.cpu()  # Boxes object for bbox outputs
        class_pred = np.array(boxes)

    cv2.imshow('Webcam', frame)

    # Calculate the cumulative sum if a denomination is detected
    
    '''
    check if classs is not none and grab the last element in the bbox which is the class element
    additionally add the banknotes if multiple are seen within the frame
    '''
    if class_pred is not None and class_pred.shape[0] > 0 and class_pred.shape[1] > 4:
        predicted_class = class_pred[-1, -1]
        if last_predicted_class != predicted_class:
            cumulative_sum += value_mappings.get(int(predicted_class), 0)
            last_predicted_class = predicted_class
            last_detection_time = time.time()
        value_text = f"Sum: {cumulative_sum}"
    else:
        current_time = time.time()
        if current_time - last_detection_time >= reset_timeout:
            cumulative_sum = 0
            last_predicted_class = None
        value_text = "No value available"
    cv2.putText(frame, value_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Value Display', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




