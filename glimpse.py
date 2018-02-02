import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json

from dynamic_fc import DynamicFC

'''
Reference: https://arxiv.org/abs/1610.04325
Instead of using just one soft attention, Glimpse computes attention
across multiple glimpses on the feature map
'''
class Glimpse(nn.Module):

	'''
	Arguments:
		glimpse_cnt = no. of glimpses on the feature map
		glimpse_embedding_size = size of the glimpse embedding
		keep_dropout = dropout probability to use
	'''
	def __init__(self, glimpse_cnt, glimpse_embedding_size, keep_dropout):
		super(Glimpse, self).__init__()

		self.glimpse_cnt = glimpse_cnt
		self.glimpse_embedding_size = glimpse_embedding_size
		self.keep_dropout = keep_dropout

		self.fc = DynamicFC().cuda()
		self.dropout = nn.Dropout(p=self.keep_dropout).cuda()
		self.softmax = nn.Softmax2d().cuda()

		self.batch_size = None
		self.channels = None
		self.height = None
		self.width = None

	'''
	Arguments: 
		feature_maps : feature_maps from the block 4 of resnet (change as required)
		context : lstm embedding of the question

	Returns:
		full_glimpse : soft attention on the feature_maps conditioned on 
					   the question embedding across multiple glimpses
	'''
	def forward(self, feature_maps, context):
		self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape

		context = self.dropout(context)
		projected_context = self.fc(context, out_planes=self.glimpse_embedding_size, activation='tanh', use_bias=False)
		# stack the context across the entire height and width of the image
		projected_context = torch.stack([projected_context]*self.height, dim=2)
		projected_context = torch.stack([projected_context]*self.width, dim=3)
		# premute the context to format (N,H,W,C) to give as input to DynamicFC
		projected_context = projected_context.permute(0,2,3,1).contiguous()

		feature_maps_orig = feature_maps.clone() # keep original for future use

		# premute the feature maps to format (N,H,W,C) to give as input to DynamicFC
		feature_maps = feature_maps.permute(0,2,3,1).contiguous()
		feature_maps = self.dropout(feature_maps)
		feature_maps = self.fc(feature_maps, out_planes=glimpse_embedding_size, activation='tanh', use_bias=False)

		hadamard = feature_maps * projected_context
		hadamard = self.dropout(hadamard)
		hadamard_emb = self.fc(hadamard, out_planes=self.glimpse_cnt, activation=None)

		# compute multiple soft glimpses and concatenate to form full glimpse
		glimpses = []
		for i in xrange(self.glimpse_cnt):
			emb = hadamard_emb[:,:,:,i]
			# expand across the total channels
			emb = torch.stack([emb]*self.channels, dim=1)
			# computes softmax across the spatial dimensions (H,W)
			alpha = self.softmax(emb)
			# get the soft glimpse
			soft_glimpses = feature_maps_orig * alpha
			soft_glimpses = soft_glimpses.view(self.batch_size, self.channels, self.height*self.height)
			# mean pool with alpha values as weights to get soft glimpse for each channel
			soft_glimpses = torch.sum(soft_glimpses, dim=2)

			glimpses.append(soft_glimpses)

		# concatenate to get full glimpse
		full_glimpse = torch.cat(glimpses, dim=1)

		return full_glimpse


'''
testing code
Arguments:
	- gpu_id
'''
if __name__ == '__main__':
	torch.cuda.set_device(int(sys.argv[1]))

	no_glimpse = 2
	glimpse_embedding_size = 1024
	keep_dropout = 1.0

	net = Glimpse(no_glimpse, glimpse_embedding_size, keep_dropout)
	feature_maps = Variable(torch.Tensor(4,2048,7,7).cuda())
	context = Variable(torch.Tensor(4,1024).cuda())
	out = net(feature_maps, context)
	print 'output: ', out.data.shape
