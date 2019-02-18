
#########################################
#    Main File for MNIST Prediction     #
#                                       #
#########################################

###Required installations: wget

####Packages to import
import os
from Data.download import load_data
import gzip

###Extract absolute directory path of the script
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

###Remote directory location of train and test files
remoteDir = "http://yann.lecun.com/exdb/mnist/"
urls = ["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]
for i in range(0,len(urls)):
    urls[i] = remoteDir + "/" + urls[i]

###Create local filenames to copy these files into, all files will be in the "Data" subfolder
localDir = script_dir + "/Data"
filenames = ["Train.gz","Train_Labels.gz","Test.gz","Test_Labels.gz"]
for i in range(0,len(filenames)):
    filenames[i] = localDir + "/" + filenames[i]


###Download the data
load_data(filenames,urls)



####Unzip files and create training, development, testing split (Note total filesize is small enough to fit in memory
####So we will just store these directly into a pandas dataframe, FOR LARGER IMAGE SIZES THIS NEEDS ADJUSTMENT

def load_all_images(file):
    import struct
    ##Read the first byte (magic number)
    magic = struct.unpack('>i',f.read(4))[0]

    #Magic number should equal 2051 if read properly
    assert(magic==2051)

    ##Read number of images
    numImages = struct.unpack('>i',f.read(4))[0]

    ###Read number of rows in each image
    rows =  struct.unpack('>i',f.read(4))[0]


    ##Read number of columns in each image
    cols =  struct.unpack('>i',f.read(4))[0]


    allImages = []
    for i in range(0,numImages):
        image = []
        for j in range(0,rows):
            format = '>' + ('B'*cols)
            currRow = struct.unpack(format,f.read(cols))
            image.append(currRow)
        allImages.append(image)

with gzip.open("Data/Train.gz","rb") as f:
    load_all_images(f)

