#!/usr/bin/python

import cv2, os
import numpy as np
from PIL import Image
from sklearn import preprocessing
from random import shuffle
import copy
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=np.nan)  #para printar o vetor inteiro


def train_and_evaluate (image_paths, image_eval_paths, k):
	# Open a file
	images = []
	eval_images = []

	#leitura da base de dados
	for image_path in image_paths:
	    # Read the image and convert to grayscale
		image_pil = Image.open(image_path).convert('L')

	    # Convert the image format into numpy array
		image = np.array(image_pil, 'int')
		image = image.reshape(-1,1)
		images.append(image)


	#leitura das imagens de teste
	for eval_path in image_eval_paths:
		# Read the image and convert to grayscale
		image_pil = Image.open(eval_path).convert('L')

	    # Convert the image format into numpy array
		image = np.array(image_pil, 'int')
		image = image.reshape(-1,1)
		eval_images.append(image)


	#transformacao das imagens em arrays de 1xN
	image_matrix = np.stack(images, axis=0)
	image_matrix = np.squeeze(image_matrix)
	eval_matrix = np.stack(eval_images, axis=0)
	eval_matrix = np.squeeze(eval_matrix)

	#calculo da media
	mean_image = np.mean( image_matrix, dtype=int, axis=0)


	cv2.imwrite ("./orl_mean.png", mean_image.reshape(112,92))

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
	eigen_faces_visualized = eigen_faces.reshape(112,92,-1)


	#saida antes de normalizar (para visualizacao)
	cv2.imwrite ("./orl_eigenface0.png", eigen_faces_visualized[:,:,0])
	cv2.imwrite ("./orl_eigenface1.png", eigen_faces_visualized[:,:,1])
	cv2.imwrite ("./orl_eigenface2.png", eigen_faces_visualized[:,:,2])
	cv2.imwrite ("./orl_eigenface3.png", eigen_faces_visualized[:,:,3])
	cv2.imwrite ("./orl_eigenface4.png", eigen_faces_visualized[:,:,4])


	#normalizando as auto-faces
	eigen_faces = eigen_faces[:,0:5]
	eigen_faces = preprocessing.normalize(eigen_faces, norm='l2')
	eigen_faces = (eigen_faces + 1) / 2


	#multiplicando as imagens pelas auto-faces para obter os coeficientes
	base_coef_matrix = np.matmul (image_matrix, eigen_faces)
	#print ('image_matrix x k_coef: {} x {} = {}'.format(image_matrix.shape, eigen_faces.shape, base_coef_matrix.shape))


	#subtraindo a media das imagens de teste
	for img in eval_matrix:
		img -= mean_image

	#multiplicando as imagens de teste pelas auto-faces para obter os coeficientes
	eval_coef_matrix = np.matmul (eval_matrix, eigen_faces)
	#print ('eval_matrix x k_coef: {} x {} = {}'.format(eval_matrix.shape, eigen_faces.shape, eval_coef_matrix.shape))



	#criando a matriz de distancias euclidianas
	dist_matrix = np.zeros((40,360))

	#calculando as distancias de cada imagem de teste para cada imagem da base
	index = 0
	for eval_coef_array in eval_coef_matrix:
		dist_array = np.array([])
		for base_coef_array in base_coef_matrix:
			eucl_dist = np.linalg.norm(eval_coef_array - base_coef_array)
			dist_array = np.insert(dist_array, dist_array.size, eucl_dist)
		dist_matrix[index, :] = dist_array
		index += 1


	hits = 0
	for i in range(0, len(image_eval_paths)):
		#print '{} is recognized as {}'.format(image_eval_paths[i][12:15], image_paths[np.argmin(dist_matrix[i,:])][12:15])
		if (image_eval_paths[i][12:15] == image_paths[np.argmin(dist_matrix[i,:])][12:15]):
			hits += 1
	#print np.argmin(dist_matrix[0,:])
	#print dist_matrix[0, np.argmin(dist_matrix[0,:])]
	
	accuracy = float(hits)/len(image_eval_paths)
	return accuracy


#leitura de todas as imagens
path = "./orl_faces"
dirs = []
paths_original = []
eval_list = []


for f in os.listdir(path):
	if os.path.isdir(path + '/' + f):
		dirs.append(path + '/' + f)

for d in dirs:
	subdirs = [os.path.join(d, sd) for sd in os.listdir(d)]
	
	for subdir in subdirs:
		paths_original.append(subdir)

#aleatorizando a ordem das imagens
shuffle(paths_original)
paths_original = np.array_split(paths_original, 10) #dividindo as imagens em 10 grupos


accuracies = []

for i in range(0, 10):
	paths_ = copy.copy(paths_original)
	eval_list = paths_.pop(i)
	acc = train_and_evaluate (np.stack(paths_).flatten(), eval_list, 5)
	print 'Accuracy for fold {}: {}'.format(i+1, acc)
	accuracies.append(acc)

print 'mean = {}'.format(np.mean(accuracies))
print 'standard deviation = {}'.format(np.std(accuracies))
accuracies = np.stack(accuracies)
plt.axis([-0.5, 9.5, -0.1, 1.1])
plt.errorbar(range(0,10), accuracies, np.std(accuracies), ecolor='red', marker='o')
plt.savefig('orl_plot.png', bbox_inches='tight')

