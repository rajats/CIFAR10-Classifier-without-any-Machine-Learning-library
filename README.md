# CIFAR10-Classifier-using-Python-and-Numpy
CIFAR10 classifier implemented as part of coursework of CS231n: Convolutional Neural Networks for Visual Recognition  

Two different type of architectures were used for designing the Convolutional Neural Network:  
I. This architecture is inspired from **VGG-16 Net** *[Simonyan and Zisserman, 2014]*. It is of the form:  
  1. Input 32X32 RGB image.    
  2. Convolutional Layer with 64 filters having size 3X3 and stride 1.  
  3. Spatial Batch Normalization Layer.    
  4. ReLU(Rectified Linear Unit) Layer for non-linearty.  
  5. Convolutional Layer with 64 filters having size 3X3 and stride 1.  
  6. Spatial Batch Normalization Layer.    
  7. ReLU(Rectified Linear Unit) Layer.   
  8. Max pooling layer with 2X2 filters and stride 2.  
  9. Convolutional Layer with 128 filters having size 3X3 and stride 1.  
  10. Spatial Batch Normalization Layer.    
  11. ReLU(Rectified Linear Unit) Layer for non-linearty.  
  12. Convolutional Layer with 128 filters having size 3X3 and stride 1.  
  13. Spatial Batch Normalization Layer.    
  14. ReLU(Rectified Linear Unit) Layer.  
  15. Max pooling layer with 2X2 filters and stride 2.  
  16. Fully Connected Layer with 256 hidden neurons.  
  17. Batch Normalization Layer.  
  18. Fully Connected Layer with 256 hidden neurons.  
  19. Batch Normalization Layer.  
  20. Fully Connected Layer giving the scores for 10 classes.  

# Performance  
This architecture was able to achieve best validation accuracy of 80.8% and test accuracy of 79.9%.  
  
# Improvement  
No deep learning libraries were used, more better performance can be achieved by using deep learning libraries and fine-tuning a pre-trained network, using even more deep convolutional neural networks, using data augmentation etc.  
  
# Acknowledgement  
This project uses course materials from [CS231n](http://cs231n.stanford.edu/).  
