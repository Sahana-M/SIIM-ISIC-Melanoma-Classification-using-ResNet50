# SIIM ISIC Melanoma Classification using ResNet50 Transfer learning
An implementation of model classifying Melanoma (malignant) against Benign conditions using ResNet50 Image classification model. Work under progress

### Data set
Find the dataset [here](https://www.kaggle.com/c/siim-isic-melanoma-classification)
- The dataset is a 1024x1024 sized image with around 98% records being Benign and rest being Malignant.
- The dataset is huge with around 33k high resolution images and harder to train on.
- Made use of the Kaggle's TPU facility to reduce the computation complexity. 

### ResNet50 Architecture
![Before](https://github.com/Sahana-M/Images/blob/master/ResNet50.png)

#### Beauty of ResNet50 in 3 points 
- **More layer is better** but because of the **Vanishing gradient** problem model weights of the first layer cannot be updated correctly through the backpropogation of the **error gradient** (the chain rule multiplies the error gradient values lower than 1 and by the time gradient error reaches the first layers, its value goes 
to zero

- That is the objective of Resnet : **preserve the gradient**.

- How ? Thanks to the **Idendity matrix** because *“what if we were to backpropagate through the identity function? Then the gradient would simply be multiplied 
by 1 and nothing would happen to it!”*.
