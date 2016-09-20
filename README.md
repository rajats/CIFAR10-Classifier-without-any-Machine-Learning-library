# CIFAR10-Classifier-using-Python-and-Numpy
CIFAR10 classifier implemented as part of coursework of CS231n: Convolutional Neural Networks for Visual Recognition  

Two different type of architectures were used for designing the Convolutional Neural Network:  
# Architecture 1
i. This architecture is inspired from **VGG-16 Net** *[Simonyan and Zisserman, 2014]*. It is of the form:  
  1. Input 32X32 RGB image.    
  2. Convolutional Layer with 64 filters having size 3X3 and stride 1.  
  3. Spatial Batch Normalization.    
  4. ReLU(Rectified Linear Unit) Layer for non-linearty.  
  5. Convolutional Layer with 64 filters having size 3X3 and stride 1.  
  6. Spatial Batch Normalization.    
  7. ReLU(Rectified Linear Unit) Layer.   
  8. Max pooling layer with 2X2 filters and stride 2.  
  9. Convolutional Layer with 128 filters having size 3X3 and stride 1.  
  10. Spatial Batch Normalization.    
  11. ReLU(Rectified Linear Unit) Layer for non-linearty.  
  12. Convolutional Layer with 128 filters having size 3X3 and stride 1.  
  13. Spatial Batch Normalization.    
  14. ReLU(Rectified Linear Unit) Layer.  
  15. Max pooling layer with 2X2 filters and stride 2.  
  16. Fully Connected Layer with 256 hidden neurons.  
  17. Batch Normalization.  
  18. Dropout of 50%.  
  19. Fully Connected Layer with 256 hidden neurons.  
  20. Batch Normalization. 
  21. Dropout of 50%.
  22. Fully Connected Layer giving the scores for 10 classes. 
  23. Softmax Layer.  

# Performance  
This architecture was able to achieve best validation accuracy of 82.7% and test accuracy of 81.1%.  
  
# Architecture 2  
ii. This architecture is of form:    
  1. Input 32X32 RGB image.      
  2. Convolutional Layer with 64 filters having size 3X3 and stride 1.    
  3. Spatial Batch Normalization.      
  4. ReLU(Rectified Linear Unit) Layer for non-linearty.    
  5. Max pooling layer with 2X2 filters and stride 2.    
  6. Convolutional Layer with 64 filters having size 3X3 and stride 1.        
  7. Spatial Batch Normalization.    
  8. ReLU(Rectified Linear Unit) Layer for non-linearty.    
  9. Max pooling layer with 2X2 filters and stride 2.    
  10. Convolutional Layer with 128 filters having size 3X3 and stride 1.          
  11. Spatial Batch Normalization.    
  12. ReLU(Rectified Linear Unit) Layer for non-linearty.   
  13. Max pooling layer with 2X2 filters and stride 2.    
  14. Convolutional Layer with 128 filters having size 3X3 and stride 1.   
  15. Spatial Batch Normalization.   
  16. ReLU(Rectified Linear Unit) Layer for non-linearty.    
  17. Fully Connected Layer with 256 hidden neurons.      
  18. Batch Normalization.   
  19. Dropout of 50%.  
  20. Fully Connected Layer with 256 hidden neurons.      
  21. Batch Normalization.   
  22. Dropout of 50%.    
  23. Fully Connected Layer giving the scores for 10 classes.  
  24. Softmax Layer.  

# Performance
This architecture was able to achieve best validation accuracy of 79.5% and test accuracy of 77.6%.  

# Improvement  
No deep learning libraries were used, more better performance can be achieved by using deep learning libraries and fine-tuning a pre-trained network, using even more deep convolutional neural networks, using data augmentation etc.  
  
# Acknowledgement  
This project uses course materials from [CS231n](http://cs231n.stanford.edu/).  
