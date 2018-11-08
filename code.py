from PIL import Image
from loopy_bp import BinaryGraph
import numpy as np
import itertools as itr 
import math

EPSILON = 0.01

def get_sample_means_covariances(flower_values, background, foreground):

	background_image = Image.open(background)
	background_values = np.array(list(background_image.getdata()))

	foreground_image = Image.open(foreground)
	foreground_values = np.array(list(foreground_image.getdata()))

	#sample averages 
	zero_indices = np.where(background_values == 1)[0]
	one_indices = np.where(foreground_values == 1)[0]

	sample_mean_1 = [0 for _ in range(3)]
	for i in one_indices:
		current = flower_values[i]
		for j in range(3):
			sample_mean_1[j] += current[j]

	sample_mean_1 = [x/(len(one_indices) - 1) for x in sample_mean_1]

	sample_variance_1 = [0 for _ in range(3)]
	for i in one_indices:
		current = flower_values[i]
		for j in range(3):
			sample_variance_1[j] += (current[j] - sample_mean_1[j])**2
	sample_variance_1 = [x/(len(one_indices) - 1) for x in sample_variance_1]

	sample_variance_1 = np.diag(sample_variance_1)

	sample_mean_0 = [0 for _ in range(3)]
	for i in zero_indices:
		current = flower_values[i]
		for j in range(3):
			sample_mean_0[j] += current[j]

	sample_mean_0 = [x/(len(zero_indices) - 1) for x in sample_mean_0]

	sample_variance_0 = [0 for _ in range(3)]
	for i in zero_indices:
		current = flower_values[i]
		for j in range(3):
			sample_variance_0[j] += (current[j] - sample_mean_0[j])**2
	sample_variance_0 = [x/(len(zero_indices) - 1) for x in sample_variance_0]

	sample_variance_0 = np.diag(sample_variance_0)

	return sample_mean_1, sample_mean_0, sample_variance_1, sample_variance_0

def create_node_edge_potentials(nodes, edges, values, sample_mean_1, sample_mean_0, sample_variance_1, sample_variance_0):

	node_potentials = {}

	determinant_1 = np.linalg.det(sample_variance_1)
	determinant_0 = np.linalg.det(sample_variance_0)
	
	print("Creating node potentials")
	for i in nodes:

		exp_value_0 = np.matmul((values[i] - sample_mean_0).transpose(), np.linalg.inv(sample_variance_0))
		exp_value_0 = np.matmul(exp_value_0, values[i] - sample_mean_0)
		exp_value_0 = np.e**(-0.5*exp_value_0)
		exp_value_0 = exp_value_0/((2*np.pi)**(1.5))
		exp_value_0 = exp_value_0/((determinant_0)**(0.5))
		exp_value_0 += EPSILON


		exp_value_1 = np.matmul((values[i] - sample_mean_1).transpose(), np.linalg.inv(sample_variance_1))
		exp_value_1 = np.matmul(exp_value_1, values[i] - sample_mean_1)
		exp_value_1 = np.e**(-0.5*exp_value_1)
		exp_value_1 = exp_value_1/((2*np.pi)**(1.5))
		exp_value_1 = exp_value_1/((determinant_1)**(0.5))
		exp_value_1 += EPSILON

		node_potentials[i] = np.array([exp_value_0, exp_value_1])
	print("Done node potentials")
	print("Creating edge potentials")
	edge_potentials = {edge: np.array([[0.9, 0.1], [0.1, 0.9]]) for edge in edges}
	print("Done edge potentials") 

	return node_potentials, edge_potentials


def image_to_graph(flower_image, flower_values): 

	width, height = flower_image.size

	nodes = set() 
	edges = set() 
	values = {}
	for i in range(height):
		for j in range(width):
		   here = (i, j) 
		   nodes.add(here) 
		   values[here] = flower_values[i*width + j]
		   if i < height - 1: 
		       down = (i + 1, j) 
		       edges.add((here, down)) 
		   if i > 0:
		   	   up = (i - 1, j)
		   	   edges.add((here, up))
		   if j < width - 1: 
		       right = (i, j + 1) 
		       edges.add((here, right)) 
		   if j > 0:
		   	   left = (i, j - 1)
		   	   edges.add((here, left))
	return nodes, edges, values






		




if __name__ == '__main__':

	flower_image = Image.open('images/flower.bmp')
	flower_values = np.array(list(flower_image.getdata()))

	nodes, edges, values = image_to_graph(flower_image, flower_values)
	sample_mean_1, sample_mean_0, sample_variance_1, sample_variance_0 = get_sample_means_covariances(flower_values, 'images/background.bmp', 'images/foreground.bmp')

	node_potentials, edge_potentials = create_node_edge_potentials(nodes, edges, values, sample_mean_1, sample_mean_0, sample_variance_1, sample_variance_0)

	g = BinaryGraph(nodes, edges, node_potentials, edge_potentials) 
	marginals = g.run_bp() 
	print(marginals)

	
	
	



