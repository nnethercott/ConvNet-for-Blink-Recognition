# ConvNet-for-Blink-Recognition

This repo includes scripts used to develop a 1-D convolutional neural network (CNN) capable of identifying blinks from raw
EEG signals detected from the FP1 and FP2 positions.  This work was done to aid the Queen's University Merlin Neurotech team 
(https://github.com/merlin-neurotech) accomplish the goal of blink recognition and some of the functions included were in part
a group effort with Derek Zhang, Awni Altabaa and Umer Kamran. The original dataset used for training was obtained from 
https://github.com/meagmohit/BLINK. 

Without significant data augmentation, the network acheived a peak accuracy of 88% when trained on 3500 inputs corresponding to
data collected over 500ms intervals.  In the future more dynamic models can easily be created by varying this time interval when
preparing the data for training. 
 
 
 Original Datasets: 
 Eye Blink Detection By Mohit Agarwal, Raghupathy Sivakumar. 57th Annual Allerton Conference on Communication, 
 Control, and Computing (Allerton). IEEE, 2019. 
