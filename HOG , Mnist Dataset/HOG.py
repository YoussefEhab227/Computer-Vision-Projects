import numpy as np
#import tensorflow as tf
#from sklearn import svm
import pandas as pd



training_data = pd.read_csv (r'C:\Users\Youssef\Desktop\Assignment 1\mnist_train.csv')
test_data = pd.read_csv (r'C:\Users\Youssef\Desktop\Assignment 1\mnist_test.csv')

xlabel = training_data["label"]
xtlabel = test_data["label"]
training_data.drop('label',
  axis='columns', inplace=True)
test_data.drop('label',
  axis='columns', inplace=True)

xtest= test_data.to_numpy()
#xtrain= data [:]

#print (xtrain[0])

cell = [8, 8]
incr = [8,8]
bin_num = 8
im_size = [32,32]


def getImage(data_list, imageIndex):
    # reading 28x28 image
    imageBefore = np.array(data_list.values[imageIndex], dtype=float).reshape(28, 28)
    # adding 4 more columns right and 4 more rows down to make it 32x32
    zeros = np.zeros(28)
    imageAfter = np.vstack([imageBefore, zeros])
    imageAfter = np.vstack([imageAfter, zeros])
    imageAfter = np.vstack([imageAfter, zeros])
    imageAfter = np.vstack([imageAfter, zeros])
    zeros2 = np.zeros(32)
    imageAfter = np.column_stack((imageAfter, zeros2))
    imageAfter = np.column_stack((imageAfter, zeros2))
    imageAfter = np.column_stack((imageAfter, zeros2))
    imageAfter = np.column_stack((imageAfter, zeros2))
    return imageAfter


for i in range (0,60000):
  training_data[i] = getImage(training_data,i)

xtrain = training_data.to_numpy()
xtrain= xtrain.reshape(60000,32,32)

def create_grad_array(image_array, index):

	#image_array = getImage(image_array,index)
	image_array = np.asarray(image_array,dtype=float)
	

	max_h = 32
	max_w = 32

	grad = np.zeros([max_h, max_w])
	mag = np.zeros([max_h, max_w])
	for h,row in enumerate(image_array):
		for w, val in enumerate(row):
			if h-1>=0 and w-1>=0 and h+1<max_h and w+1<max_w:
				dy = image_array[h+1][w]-image_array[h-1][w]
				dx = row[w+1]-row[w-1]+0.0001
				grad[h][w] = np.arctan(dy/dx)
				if grad[h][w]<0:
					grad[h][w] += 180
				mag[h][w] = np.sqrt(dy*dy+dx*dx)
	
	return grad,mag





def calculate_histogram(array,weights):
	bins_range = (0, 180)
	bins = bin_num
	hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)

	return hist


def create_hog_features(grad_array,mag_array):
	max_h = int(((grad_array.shape[0]-cell[0])/incr[0])+1)
	max_w = int(((grad_array.shape[1]-cell[1])/incr[1])+1)
	cell_array = []
	w = 0
	h = 0
	i = 0
	j = 0

	#Creating 8X8 cells
	while i<max_h:
		w = 0
		j = 0

		while j<max_w:
			for_hist = grad_array[h:h+cell[0],w:w+cell[1]]
			for_wght = mag_array[h:h+cell[0],w:w+cell[1]]
			
			val = calculate_histogram(for_hist,for_wght)
			cell_array.append(val)
			j += 1
			w += incr[1]

		i += 1
		h += incr[0]

	cell_array = np.reshape(cell_array,(max_h, max_w, bin_num))
	#normalising blocks of cells
	block = [2,2]
	#here increment is 1

	max_h = int((max_h-block[0])+1)
	max_w = int((max_w-block[1])+1)
	block_list = []
	w = 0
	h = 0
	i = 0
	j = 0

	while i<max_h:
		w = 0
		j = 0

		while j<max_w:
			for_norm = cell_array[h:h+block[0],w:w+block[1]]
			mag = np.divide(for_norm, np.sum(for_norm))
			arr_list = (mag).flatten().tolist()
			block_list += arr_list
			j += 1
			w += 1

		i += 1
		h += 1

	#returns a vector array list of 288 elements
	return block_list



def apply_hog(image_array,index):
	gradient,magnitude = create_grad_array(image_array,index)
	hog_features = create_hog_features(gradient,magnitude)
	hog_features = np.asarray(hog_features,dtype=float)
	hog_features = np.expand_dims(hog_features,axis=0)

	return hog_features

x_train = np.zeros((60000,784), dtype=float)
for i in range(0,60000):
        x = apply_hog(xtrain[i],i)
      #  x = np.resize(x,(1,784))
        x_train[i] = x




from sklearn.svm import SVC
print('IN')
model = SVC(gamma='auto')
model.fit(x_train , xlabel)
y_predicted = model.predict(xtest[:288,:])
from sklearn import metrics
print("Accuracy: " , metrics.accuracy_score(xtlabel,y_predicted))