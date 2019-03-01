# How to specify a new model!

In order to specify a model, the model parser accepts files in the following format.

The first line must consist of a single line which reads "Layers:"

Next, the file will have N lines consisting of components of a model architecture. The full options are:

### 2D Convolution
For a 2D convolutional layer, the syntax is

Conv,\<num-filters>,\<kernel-rows>,\<kernel-cols>,\<activation-func>
  
Ensure the activation function is a correctly specified activation as required by Keras.

### Max Pooling
For a max pooling layer, the syntax is

MaxPool,\<kernel-rows>,\<kernel-cols>
  
### Dropout

For a dropout regularization of the previous layer, the syntax is

Dropout,\<dropout-proportion>
  
### Flatten
To flatten a the convolutional layers for preparation as input to the fully connected layers, the syntax is just:

Flatten

### Fully Connected Layers

For fully connected layers the syntax is:

Dense,\<num-neurons>,\<activation-func>
  
Again, be sure the activation function is a valid Keras activation

### Optimizer and Training Params
Once the model architecture is fully specified the syntax for the final lines of the file consist of:

Optimizer:
\<learning-rate>,\<decay>
\<num-epochs>
\<batch-size>
  
Note that currently only Adam optimization is being used. Feel free to expand the ModelParser to improve functionality!

### For help, consult the example_model.txt file! All model files should be saved in the Models folder.

