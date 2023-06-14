# visual_money_counter


A computer vision project for the detection of banknote denomination

# Table of Contents
- [About The Project](#About-the-project)
- [Getting Started](#getting-started)
- [License](#license)

## About The Project

The project aims to to classify the banknote denomination in realtime using the webcam,additionally show the sum if multiple banknote values are on the screen

Image annoations and augmentations are done through 
- python scripts
- makesense
 
The model is trained and built with the help of YOLOv8 specifically the YOLOv8n and yolov8m , the number of annotated images are 1k and will be increased soon to help improve its accuracy and consistency.

Some examples of the results

![results_1](https://github.com/Logatta/visual_money_counter/assets/127979606/e1c93e7f-cafc-4619-b42f-b9e0562b8091)
  
## Getting started 
instructions for setting up the detection model 

Prerequisites

1.the following must be available
- python 3 or above
- Source to stream video (webcam)

2.Install the requirments 
'''python
print('pip install -r requirmnets.txt')

3.run this following command 
python detect.py --weights runs/model_trained/bank_notes_v2.pt --source 0


## License

Distributed under the MIT License. See LICENSE.txt for more information.

