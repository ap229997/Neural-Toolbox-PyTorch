import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json

from dynamic_fc import DynamicFC

'''
Computes attention mechanism on the image feature_maps (N,C,H,W) conditioned on the question embedding
'''
class Attention(nn.Module):

	def __init__(self, config):
		super(Attention, self).__init__()

		self.mlp_units = config['model']['image']['attention']['no_attention_mlp'] # hidden layer units of the MLP

		# MLP for concatenated feature_maps and question embedding
		# input channels = 3072 : feature_maps (N,2048,H,W) + question embedding (N,1024)
		# change input channels as required - define the dimensions in config file
		self.fc = nn.Sequential(
            nn.Linear(3072, self.mlp_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_units, 1),
            nn.ReLU(inplace=True),
            ).cuda()

		############ ALTERNATIVE ############
		# use dynamic MLP module and specify the input, output planes, activation fn and whether to use bias on the go
		# use nn.Sequential to process multiple dynamic MLPs sequentially
		# self.fc = DynamicFC().cuda()

		self.softmax = nn.Softmax2d() # to get the probablity values across the height and width of feature_maps

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight)
				nn.init.constant(m.bias, 0.1)

		self.batch_size = None
		self.channels = None
		self.height = None
		self.width = None
		self.len_emb = None

	'''
	Arguments: 
		feature_maps : feature_maps from the block 4 of resnet (change as required)
		context : lstm embedding of the question

	Returns:
		soft_attention : soft attention on the feature_maps conditioned on the question embedding
	'''
	def forward(self, feature_maps, context):
		self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape
		self.len_emb = context.data.shape[1]

		feature_maps = feature_maps.view(self.batch_size, self.channels, self.height*self.width)

		context = torch.stack([context]*self.height*self.width, dim=2)

		embedding = torch.cat([feature_maps, context], dim=1)
		embedding = embedding.permute(0,2,1).contiguous() # permute concatenated embedding according to MLP dimensions

		out = self.fc(embedding)
		out = out.permute(0,2,1).contiguous()
		out = out.view(self.batch_size, 1, self.height, self.width)
		# get the probability values across the height and width of the feature_maps
		alpha = self.softmax(out)

		feature_maps = feature_maps.view(self.batch_size, self.channels, self.height, self.width)
		soft_attention = feature_maps * alpha
		soft_attention = soft_attention.view(self.batch_size, self.channels, -1)
		# mean pool across the height and width of the feature_maps with alpha value serving as weights
		soft_attention = torch.sum(soft_attention, dim=2)

		return soft_attention

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

	net = Attention(config)
	feature_maps = Variable(torch.Tensor(4,2048,7,7).cuda())
	context = Variable(torch.Tensor(4,1024).cuda())
	out = net(feature_maps, context)
	print 'output: ', out.data.shape
