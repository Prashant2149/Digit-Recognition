import matplotlib.pyplot as plt
from sklearn import datasets,svm
digits=datasets.load_digits()  #here it is not as basic dictionary it is a dataframe
print "digits:",digits.keys()
print "digits.target-----------",digits.target


images_and_labels=list(zip(digits.images,digits.target))
print "len(images_and_labels)",len(images_and_labels)

for index,[image,label] in enumerate(images_and_labels[:5]): #for black 0 and for white it is max
    print "index:",index,"image:\n",image, "  Label:",label             #Thus we are cutting in suh a way that we are getting data in three parts as array Index ,Image and Target
    plt.subplot(2,5,index+1) #Position numbering starts from 1
    plt.axis('on') #used to on the ticks
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest') #used to show the Image

    plt.title('Training %i'%label)
    #Image is multidimensional matrix and the intensity is represented in it
#plt.show()


n_samples=len(digits.images)
print "n_samples :",n_samples

imageData=digits.images.reshape((n_samples,-1)) #Here image was of 2 D and it is being converted to 1 D
print "After reshaped : len(imageData[0]):",len( imageData[0])

#create a Classifies : a support vector classifier
classifier=svm.SVC(gamma=0.001) #learning rate should always be small and represented by gamma
#We learn the digits on the first half of the digits
classifier.fit(imageData[ :n_samples//2],digits.target[ :n_samples//2])
expected=digits.target[n_samples//2: ]
predicted=classifier.predict(imageData[n_samples//2:]) #here we will apply the Mahine Learning so we will use the Processed Data
#imagedata being calculated can't be printed on screen as it's data has been processed an d can be used in calculation
#But if we want to print the first row of image data then probably it will be used only in Machine Learning
#and the data is being obtained as digits.images
#but to fit we use the processed image data for training purpose

images_and_predictions=list(zip(digits.images[n_samples//2:],predicted))
for index,[image,prediction] in enumerate(images_and_predictions[:5]):
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction:%i' % prediction)
print 'Original Values :',digits.target[n_samples//2: (n_samples//2)+5]
plt.show()
classifier=svm.SVC(gamma=0.001)
classifier.fit(imageData[:],digits.target[:])
#used to process live image resize it to 8*8
from scipy.misc import imread,imresize,bytescale

img=imread("th.jpeg")
img=imresize(img,(8,8)) #changes it to 8*8 images
img=img.astype(digits.images.dtype) #changes it to datatype needed i.e. Pandas
img=bytescale(img,high=16.0,low=0) #resolution change to 0 to 16 as same resolution
#in training of images
print " img :======\n",img.shape, "\n",img #Here we have changed it to 8*8 and total made of 3 colours so [8,8,3]
x_testData=[]
for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0)
print "x_testData :\n",x_testData

print "len(x_testData):\n",len(x_testData)
x_testData=[x_testData]
print"len(x_testData):\n",len(x_testData)
print"Machine output=",classifier.predict(x_testData)
plt.show()
