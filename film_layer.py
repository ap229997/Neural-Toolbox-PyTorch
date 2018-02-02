import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json

from dynamic_fc import DynamicFC

'''
appends spatial co-ordinates to the feature map
the range of the spatial co-ordinates can be defined as [min_val, max_val]

Arguments:
	feature_maps : input feature maps (N,C,H,W)
	min_val : minimum value of spatial co-ordinate
	max_val : maximum value of spatial co-ordinate

Returns:
	feature_maps : input feature maps concatenated with spatial co-ordinates (N,C+2,H,W) 
'''
def append_spatial_location(feature_maps, min_val=-1, max_val=1):
	batch_size, channels, height, width = feature_maps.data.shape

	# arange spatial co-ordiantes for height
	h_array = Variable(torch.stack([torch.linspace(min_val, max_val, height)]*width, dim=1).cuda())
	# arange spatial co-ordinates for width
	w_array = Variable(torch.stack([torch.linspace(min_val, max_val, width)]*height, dim=0).cuda())
	# stack the h_array and w_array to get the spatial co-ordinates at each location
	spatial_array = torch.stack([h_array, w_array], dim=0)
	# expand the spatial co-ordinates across the batch size
	spatial_array = torch.stack([spatial_array]*batch_size, dim=0)
	# concatenate feature maps with spatial co-ordinates
	feature_maps = torch.cat([feature_maps, spatial_array], dim=1)

	return feature_maps


'''
A very basic FiLM layer with a linear transformation from context to FiLM parameters
'''
class FilmLayer(nn.Module):
	
	def __init__(self):
		super(FilmLayer, self).__init__()

		self.batch_size = None
		self.channels = None
		self.height = None
		self.width = None
		self.feature_size = None

		self.fc = DynamicFC().cuda()

	'''
	Arguments:
		feature_maps : input feature maps (N,C,H,W)
		context : context embedding (N,L)

	Return:
		output : feature maps modulated with betas and gammas (FiLM parameters)
	'''
	def forward(self, feature_maps, context):
		self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape
		# FiLM parameters needed for each channel in the feature map
		# hence, feature_size defined to be same as no. of channels
		self.feature_size = feature_maps.data.shape[1]

		# linear transformation of context to FiLM parameters
		film_params = self.fc(context, out_planes=2*self.feature_size, activation=None)

		# stack the FiLM parameters across the spatial dimension
		film_params = torch.stack([film_params]*self.height, dim=2)
		film_params = torch.stack([film_params]*self.width, dim=3)

		# slice the film_params to get betas and gammas
		gammas = film_params[:, :self.feature_size, :, :]
		betas = film_params[:, self.feature_size:, :, :]

		# modulate the feature map with FiLM parameters
		output = (1 + gammas) * feature_maps + betas

		return output

'''
Modualted ResNet block with FiLM layer 
'''
class FilmResBlock(nn.Module):

	'''
	Arguments:
		in_channels : no. of channels in the input
		feature_size : feature size required
		spatial_location : whether to append spatial co-ordinates
	'''
	def __init__(self, in_channels, feature_size, spatial_location=True):
		super(FilmResBlock, self).__init__()

		self.spatial_location = spatial_location
		self.feature_size = feature_size
		self.in_channels = in_channels
		# add 2 channels for spatial location
		if spatial_location:
			self.in_channels += 2

		# modulated resnet block with FiLM layer
		self.conv1 = nn.Conv2d(self.in_channels, self.feature_size, kernel_size=1)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(self.feature_size)
		self.film_layer = FilmLayer().cuda()
		self.relu2 = nn.ReLU()

	'''
	Arguments:
		feature_maps : input feature maps (N,C,H,W)
		context : context embedding (N,L)

	Returns:
		out : input feature maps modulated with FiLM parameters computed using context embedding
	'''
	def forward(self, feature_maps, context):

		if self.spatial_location:
			feature_maps = append_spatial_location(feature_maps)

		conv1 = self.conv1(feature_maps)
		out1 = self.relu1(conv1)

		conv2 = self.conv2(out1)
		bn = self.bn2(conv2)
		film_out = self.film_layer(bn, context)
		out2 = self.relu2(film_out)
		
		# residual connection
		out = out1 + out2

		return out

'''
testing code
Arguments:
	- gpu_id
	- config file
'''
if __name__ == '__main__':
	torch.cuda.set_device(int(sys.argv[1]))

	with open(sys.argv[2], 'rb') as f_config:
		config_str = f_config.read()
		config = json.loads(config_str.decode('utf-8'))

	net = FilmLayer()
	feature_maps = Variable(torch.Tensor(4,2048,7,7).cuda())
	context = Variable(torch.Tensor(4,1024).cuda())
	out = net(feature_maps, context)
	print 'output: ', out.data.shape
	out = append_spatial_location(feature_maps)
	print 'output: ', out.data.shape
	net2 = FilmResBlock(2048, 2048).cuda()
	out = net2(feature_maps, context)
	print 'output: ', out.data.shape
