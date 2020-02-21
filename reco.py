import struct
import numpy
import numpy as np
import  Kmeans
from matplotlib import style
style.use('ggplot')

trainingImage=open("./dataset/train-images-idx3-ubyte",'rb')
MSB = struct.unpack('>4B', trainingImage.read(4)) ##reading most signficant bit in big indian
print(MSB)
numberOfImages = struct.unpack('>I', trainingImage.read(4))[0] ##reading number of images
print("number of images : " ,numberOfImages)
numberOfRows = struct.unpack('>I', trainingImage.read(4))[0] ##reading number of images
print("number of rows : " ,numberOfRows)
numberOfCols = struct.unpack('>I', trainingImage.read(4))[0] ##reading number of images
print("number of Cols : " ,numberOfCols)
#Reading image as array
images = numpy.asarray(struct.unpack('>' + 'B'*numberOfImages*numberOfRows*numberOfCols, trainingImage.read(numberOfImages*numberOfRows*numberOfCols)))
#reshape images to 60000 matrix 28*28
imagesMatrix = numpy.reshape(images, (numberOfImages, numberOfRows, numberOfCols))/255

trainX = imagesMatrix
#################################### model
colors = 10*["g","r","c","b","k"]
print(imagesMatrix[0])

data = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
                 [1, 3],
                 [8, 9],
                 [0, 3],
                 [5, 4],
                 [6, 4]
                 ])


kmeans= Kmeans.K_Means()
y=[]
kmeans.train(trainX)




#clf = test.K_Means()
#clf.fit(imagesMatrix)
#trainingLabel=open("/home/ziad/PycharmProjects/HandWrittenDigits/dataset/train-labels-idx1-ubyte",'rb')
#MSB = struct.unpack('>4B', trainingLabel.read(4)) ##reading most signficant bit in big indian
#print(MSB)
#numberOfItems = struct.unpack('>I', trainingLabel.read(4))[0] ##reading number of items
#print(numberOfItems)
#labels = numpy.asarray(struct.unpack('>' + 'B'*numberOfItems, trainingLabel.read(numberOfItems)))
#print(labels)