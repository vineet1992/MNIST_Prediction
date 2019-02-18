

###Function to download training and testing data to the Data folder

def load_data(filenames,urls):
    import wget
    for i in range(0,len(filenames)):
        wget.download(urls[i],out=filenames[i])