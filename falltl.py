
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
from scipy import ndimage, misc
from keras.utils import np_utils

path = 'C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/imagecolor/'
#path = 'C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/Dataset and Labels/new Dataset and Lables/imagenew2/'
SSS=206
data = np.zeros((SSS,64,68,3))
zz=[]

for i in range(SSS):    
    zz.append(str(i+1)+".jpg")

for ii, imagee in enumerate(zz):
    path2 = os.path.join(path, imagee)
    image2 = ndimage.imread(path2)
    image2=image2.astype(np.float64)
    image2= scipy.misc.imresize(image2, 0.5)
#    data2 = img_to_array(image2)
#    data2=np.squeeze(data2)
#    data = image3.reshape(129*139, 1)
#    data[ii,:,:]=image2/255
    data[ii,:,:,:]=image2/255

import csv
with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     label=np.squeeze(label) 
     
#X_train=data[:158,:,:,:]
#X_test=data[158:,:,:,:]
#y_train=label[:158]
#y_test=label[158:]
m=0

kf=KFold(3, random_state=None, shuffle=False)
kf.get_n_splits(data)
k=0
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    if k==m:
       break 
    k=k+1
    
VV=X_train
VV1=X_test  
  
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation 
from keras import initializers, optimizers
 
nClasses=2
VV = VV.reshape(VV.shape[0], VV.shape[1], VV.shape[2], 3)
VV1 = VV1.reshape(VV1.shape[0], VV1.shape[1], VV1.shape[2], 3)
input_shape=VV.shape[1:4]
  
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
#from keras.applications.mobilenet import MobileNet
#from keras.applications.densenet import DenseNet121
#from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, normalization

# create the base pre-trained model
base_model =  VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x1 = GlobalAveragePooling2D()(x)
#x1 = Flatten(input_shape=x.shape[1:])(x)
# let's add a fully-connected layer
#x1 = Dense(1024, activation='relu')(x)
#x1 = normalization.BatchNormalization()(x1)
predictions = Dense(2, activation='softmax')(x1)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
for layer in model.layers[:20]:########VGG16
   layer.trainable = False
for layer in model.layers[20:]:
   layer.trainable = True
#for layer in model.layers[:20]:########VGG19
#   layer.trainable = False
#for layer in model.layers[20:]:
#   layer.trainable = True
#for layer in model.layers[:153]:########ResNet
#   layer.trainable = False
#for layer in model.layers[153:]:
#   layer.trainable = True

model.summary()
batch_size = 16
epochs =200 
opti=optimizers.SGD(lr=0.001, momentum=0.9) #Adam(lr=0.0002) #0.0002   #
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model.fit(VV, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(VV1, Y_test))

model.evaluate(VV1, Y_test, verbose=0) 
y_pred = model.predict(VV1)          

preds = np.argmax(y_pred, axis=1)

## Loss Curves
#plt.figure(figsize=[8,6])
#plt.plot(history.history['loss'],'r',linewidth=3.0)
#plt.plot(history.history['val_loss'],'b',linewidth=3.0)
#plt.legend(['Training loss', 'Test Loss'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Loss',fontsize=16)
#plt.title('Loss Curves',fontsize=16)
# 
## Accuracy Curves
#plt.figure(figsize=[8,6])
#plt.plot(history.history['acc'],'r',linewidth=3.0)
#plt.plot(history.history['val_acc'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Test Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves',fontsize=16)

labels = {1:'Non-Fall', 2:'Fall'}
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(preds, y_test,
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(preds, y_test)

fig = plt.figure(figsize=(2,2))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
#cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(2), [l for l in labels.values()])


