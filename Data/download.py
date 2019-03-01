import wget


###Function to download training and testing data to the Data folder

def load_data(urls,filenames):
    ###Download the files from the server
    for i in range(0,len(filenames)):
        print("Downloading file " + urls[i] + ", " + str(i+1) + " out of " + str(len(filenames)))
        temp = wget.download(urls[i],out=filenames[i])
