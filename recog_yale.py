#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
import scipy.misc
import operator
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt


def train_and_evaluate (leave_one_out, k):
	#np.set_printoptions(threshold=np.nan)  #para printar o vetor inteiro
	# For face detection we will use the Haar Cascade provided by OpenCV.
	cascadePath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath)

	# For face recognition we will the the LBPH Face Recognizer 
	recognizer = cv2.createLBPHFaceRecognizer()


	# Path to the Yale Dataset
	path = './yalefaces'

	# Append all the absolute image paths in a list image_paths
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith(leave_one_out)]
	image_eval_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(leave_one_out)]

	images = []
	eval_images = []

	for image_path in image_paths:
	    # Read the image and convert to grayscale
	    image_pil = Image.open(image_path).convert('L')
	    # Convert the image format into numpy array
	    image = np.array(image_pil, 'uint8')

	    faces = faceCascade.detectMultiScale(image)
	    for (x, y, w, h) in faces:
	    	images.append(image[y: y + h, x: x + w])


	for image_path in image_eval_paths:
		# Read the image and convert to grayscale
	    image_pil = Image.open(image_path).convert('L')
	    # Convert the image format into numpy array
	    image = np.array(image_pil, 'uint8')

	    faces = faceCascade.detectMultiScale(image)
	    for (x, y, w, h) in faces:
	    	eval_images.append(image[y: y + h, x: x + w])


	#redimensionamento das imagens
	images_to_resize = images
	images = []
	for img in images_to_resize:
		resized_image = scipy.misc.imresize (img, (170, 170)).reshape(-1,1)
		resized_image = resized_image.astype(int)
		images.append(resized_image)

	#redimensionamento das imagens de teste
	images_to_resize = eval_images
	eval_images = []
	for img in images_to_resize:
		resized_image = scipy.misc.imresize (img, (170, 170)).reshape(-1,1)
		resized_image = resized_image.astype(int)
		eval_images.append(resized_image)


	#transformacao das imagens em arrays de 1xN
	image_matrix = np.stack (images, axis=0)
	image_matrix = np.squeeze(image_matrix)
	eval_matrix = np.stack(eval_images, axis=0)
	eval_matrix = np.squeeze(eval_matrix)

	#calculo da media
	mean_image = np.mean (image_matrix, dtype=int, axis=0)

	#cv2.imwrite ("./yale_mean.png", mean_image.reshape(170,170))

	#subtraindo a media de todas as imagens
	for img in image_matrix:
		img -= mean_image

	#calculo da matriz de covariancia A.T * A
	covariance_matrix = np.matmul (image_matrix, image_matrix.T)

	#calculo dos auto-valores e auto-vetores
	eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
	eigen_vectors = eigen_vectors.real
	eigen_values = eigen_values.real


	#obtendo as auto-faces
	eigen_faces = np.matmul(image_matrix.T, eigen_vectors)
	eigen_faces_visualized = eigen_faces.reshape(170,170,-1)


	#normalizando as auto-faces
	eigen_faces = eigen_faces[:,0:k]
	eigen_faces = preprocessing.normalize(eigen_faces, norm='l2')
	eigen_faces = (eigen_faces + 1) / 2


	#multiplicando as imagens pelas auto-faces para obter os coeficientes
	base_coef_matrix = np.matmul (image_matrix, eigen_faces)


	#subtraindo a media das imagens de teste
	for img in eval_matrix:
		img -= mean_image

	#multiplicando as imagens de teste pelas auto-faces para obter os coeficientes
	eval_coef_matrix = np.matmul (eval_matrix, eigen_faces)



	#criando a matriz de distancias euclidianas
	dist_matrix = np.zeros((15, len(image_paths)))

	#calculando as distancias de cada imagem de teste para cada imagem da base
	index = 0
	for eval_coef_array in eval_coef_matrix:
		dist_array = np.array([])
		for base_coef_array in base_coef_matrix:
			eucl_dist = np.linalg.norm(eval_coef_array - base_coef_array)
			dist_array = np.insert(dist_array, dist_array.size, eucl_dist)
		dist_matrix[index, :] = dist_array
		index += 1


	#arrumando as labels
	img_labels = []
	for image_path in image_paths:
		img_labels.append(image_path[12:21])

	eval_labels = []
	for eval_path in image_eval_paths:
		eval_labels.append(eval_path[12:21])


	hits = 0
	for i in range(0, len(image_eval_paths)):
		#print '{} is recognized as {}'.format(eval_labels[i], img_labels[np.argmin(dist_matrix[i,:])])
		if (eval_labels[i] == img_labels[np.argmin(dist_matrix[i,:])]):
			hits += 1
	accuracy = float (hits)/len(eval_labels)
	print 'Accuracy when using {} for testing: {}'.format(leave_one_out, accuracy)

	return accuracy


accuracies = []

accuracies.append(train_and_evaluate ('.sad', 5))
accuracies.append(train_and_evaluate ('.centerlight', 5))
accuracies.append(train_and_evaluate ('.glasses', 5))
accuracies.append(train_and_evaluate ('.happy', 5))
accuracies.append(train_and_evaluate ('.leftlight', 5))
accuracies.append(train_and_evaluate ('.noglasses', 5))
accuracies.append(train_and_evaluate ('.normal', 5))
accuracies.append(train_and_evaluate ('.rightlight', 5))
accuracies.append(train_and_evaluate ('.sleepy', 5))
accuracies.append(train_and_evaluate ('.surprised', 5))
accuracies.append(train_and_evaluate ('.wink', 5))

print 'mean = {}'.format(np.mean(accuracies))
print 'standard deviation = {}'.format(np.std(accuracies))

accuracies = np.stack(accuracies)
plt.axis([-0.5, 10.5, -0.1, 1.1])
plt.errorbar(range(0,11), accuracies, np.std(accuracies), ecolor='red', marker='o')
plt.savefig('yale_plot.png', bbox_inches='tight')