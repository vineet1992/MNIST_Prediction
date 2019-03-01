# MNIST Digit Classification System
MNIST Digit Classification using Convolutional Neural Networks

This project presents a system for training, tuning hyperparameters, and evaluating a Convolutional Neural Network for MNIST digit classification.


# Production System for MNIST Classification

The MNIST handwritten dataset is a popular benchmark for image classification algorithms, especially CNN's. The purpose of this project is to have an enclosed system for beginners to experiment with CNN architectures on the MNIST dataset without doing any extra work. The project includes commands for downloading the data, parsing a CNN model from a text file and training the CNN, Hyperparameter exploration using Talos, and Evaluating proposed CNN's on the test set and serializing output models.

## Dependencies

The full list of dependencies for the project are laid out in the requirements.txt file. For use, please first create a python virtual enviornment on your machine, and then use: 

pip install -r requirements.txt 

Ensure that you are the main directory of the project in order to access requirements.txt

# Downloading Data

The data is downloaded directly from this link: "http://yann.lecun.com/exdb/mnist/" To do so, please use the following command

python main.py download <dataset-dir>
  
The argument specifies the name of the directory to download the data to. Please keep this consistent throughout all commands of the project

# Training a Model

# Testing a Model

# Exploring Hyperparameters

# Specifying a Model

In order to specify a model, the model parser accepts files in the following format.

The first line must consist of a single line which reads "Layers:"

Next, the file will have N lines consisting of components of a model architecture. The full options are:

### 2D Convolution
For a 2D convolutional layer, the syntax is

Conv,<num-filters>,<kernel-rows>,<kernel-cols>,<activation-func>
  
Ensure the activation function is a correctly specified activation as required by Keras.

### Max Pooling
For a max pooling layer, the syntax is

MaxPool,<kernel-rows>,<kernel-cols>
  
### Dropout

For a dropout regularization of the previous layer, the syntax is

Dropout,<dropout-proportion>
  
### Flatten
To flatten a the convolutional layers for preparation as input to the fully connected layers, the syntax is just:

Flatten

### Fully Connected Layers

For fully connected layers the syntax is:

Dense,<num-neurons>,<activation-func>
  
Again, be sure the activation function is a valid Keras activation

### Optimizer and Training Params
Once the model architecture is fully specified the syntax for the final lines of the file consist of:

Optimizer:
<learning-rate>,<decay>
<num-epochs>
<batch-size>
  
Note that currently only Adam optimization is being used. Feel free to expand the ModelParser to improve functionality!

### For help, consult the example_model.txt file! All model files should be saved in the Models folder.

 
## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

This is not an officially supported Google product.
