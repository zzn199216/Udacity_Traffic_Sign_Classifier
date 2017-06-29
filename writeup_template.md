#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./save_img/counter_lables.jpg "Visualization"
[image2]: ./save_img/ogimg.jpg "Grayscaling"
[image3]: ./save_img/grayandnorm.jpg  "Random Noise"
[image4]: ./save_img/8sign.jpg "Traffic Sign 1"
[image5]: ./save_img/8signacc.jpg "Traffic Sign 1"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I think graph is more value than color in this project.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

after I use normalize function in cv2 to make a normalize function to make normalization more convenient.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the we can see the normed image seem serrated,and original data seem more smooth.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 28x28x6	 				|
| Convolution 2x2     	| 2x2 stride, same padding, outputs 24x24x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 				|
| Convolution 2x2     	| 2x2 stride, same padding, outputs  8x8x32		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x32					|
| flatten			    | in 521 , out 256   							|
| dropout				| 0.9        									|
| flatten				| in 256 , out 86    							|
| dropout				| 0.8        									|
| flatten				| in 86 , out 43								|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an softmax_cross_entropy_with_logits to calculate logits and use AdamOptimizer with 99 epochs and a changing learning_rate,ever 10 epochs it decrease to 0.6 of it self.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.88%
* validation set accuracy of 96.67%
* test set accuracy of 94.99%

If an iterative approach was chosen:

I use the LeNet architecture at the first time. because it was in the course.

the accuracy was about 70%.might because wasn't set the data gray and the depth is to low.

I add some layer to make it more accurate and I add some dropout layer to make it less overfitting.

I tuned the learning rate because i found it went awful when epochs get highier. so I make it changing over time. and after tuned , the accuracy increase about 3%.

I think becaus1 graph means a lot in traffic sign , so convolution layer can do it good.and because there is a lot of noise like something out of the sign,make it shoud use some way like dropout to prevent overfitting.


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8  traffic signs that I found on the web:

![alt text][image4] 

The 2,3,4,7 image might be difficult to classify because even I can hardly recognize with it is.might need highier resolution.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| speed limit(80km/h)   | speed limit(80km/h)							| 
| truck     			| Stop											|
| sidewalk				| General caution								|
| General caution	    | road working						 			|
| No entry     			|  No entry  									|
| No Tooting			| Vehicles over 3.5 metric tons prohibited		|
| stop					|  stop											|
| Yield     			| Yield											|


The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. This is much low to the accuracy on the test set .might because compare to the image , the sign in it is too small, make the model hard to recognize it.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
![alt text][image5]
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Go straight or right sign (probability of 0.65), and the image does contain a Go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .65         			| Go straight or right  						| 
| .35     				| Roundabout mandatory 							|
| 0						|  												|
| 0		      			|			 									|
| 0					    | 	    										|


For the second image,the model is relatively sure that this is a Yield sign (probability of 0.65), and the image does contain a Yield sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .68         			| Yield											| 
| .3     				| Roundabout mandatory 							|
| 0.02					| speed limit(50km/h)							|
| 0		      			|			 									|
| 0					    | 	    										|

For the 3rd image,the model is relatively sure that this is a General caution(probability of 1)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| General caution								| 
| 0     				|												|
| 0						| 												|
| 0		      			|			 									|
| 0					    | 	    										|

For the 4th image,the model is relatively sure that this is a End of no passing limits (probability of 0.88)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.95         			| End of no passing								| 
| 0.04    				| Stop											|
| 0.01					| Priority Road									|
| 0		      			|			 									|
| 0					    | 	    										|

For the second image,the model is relatively sure that this is a No entry sign (probability of 1), and finally,it is.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No entry 										| 
| 0     				| 												|
| 0						| 												|
| 0		      			|			 									|
| 0					    | 	    										|
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
 I think it'e edge.

