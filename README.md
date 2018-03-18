# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Model Visualization"
[image2]: ./examples/gray_image.png "Grayscaling"
[image3]: ./examples/gray_image2.png "Recovery Image"
[image4]: ./examples/gray_image3.png "Recovery Image"
[image5]: ./examples/gray_image4.png "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model's shape is same as NVIDIA model.
My model consists of 5 convolution neural networks, 3 layers with 5x5 filter sizes, depths between 24 and 48 (model.py lines 86-90) and 2 layers with 3x3 filter sizes, depth 64(code lines 92-94).
The model includes RELU layers to introduce nonlinearity (code line 72-79), and the data is normalized in the model using a Keras BatchNoramlization layer (code line 85). Output layer use linear activation function because of regulation.(code line 102)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 87-98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 45). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Also, model used mean squared error as loss function.(model.py line 104)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a proven model, concentrate regulation, make recovery input and prevent overfitting.

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because this model is based on real autonomous driving. This model's regulation ability is already verified.

In order to gauge how well the model was working, I split my images and steering angle datas into a training and validation set. I found that result of model's loss was very low but my first automous driving was failed because of overfitting.

To combat the overfitting, I modified the model so that I added 6 dropout layers.

Then I added others input datas in different situation like moving car side to middle.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like bridge road and road with a byway to improve the driving behavior in these cases, I repeated recoding input datas about those driving parts several times for solving those problems.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 84-107) consisted of one normalized layer, 5 convolution neural networks and 3 fully connected layers. first three layers have 5x5 kernel size, 2x2 stride. First layer has 24 x 31 x 98 size. Second layer has 36 x 14 x 47 and Third layer has 48 x 5 x 22. Last two convolution layers has no stride and 3 x 3 kernel size. First layer has 64 x 3 x 20 size and second layer has 64 x 1 x 18. After then, 3 fully connected layers lead an output value.

Here is a visualization of the architecture :

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used sample datas which were provided by Udacity on track one using center lane driving. Here is an example image of center lane driving :

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to keep in center and if car go wrong direction it can fix and find right way. These images show what a recovery looks like starting from right side of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

After the collection process, I had about 24,000 number of data points. I then preprocessed this data by using python genetor each one batches in 128 datas.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by several tests. When more than 5 epochs, a little decreasing train set's loss but validation set had little change, sometimes increasing loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
