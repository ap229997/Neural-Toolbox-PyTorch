import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
import json

'''
dynamic FC - for varying dimensions on the go
Input - embedding in (batch_size(N), * , channels(C)) [* represent extra dimensions]
'''

class DynamicFC(nn.Module):

	def __init__(self):
		super(DynamicFC, self).__init__()

		self.in_planes = None
		self.out_planes = None
		self.activation = None
		self.use_bias = None

		self.activation = None
		self.linear = None

	'''
	Arguments:
		embedding : input to the MLP (N,*,C)
		out_planes : total channels in the output
		activation : 'relu' or 'tanh'
		use_bias : True / False

	Returns:
		out : output of the MLP (N,*,out_planes)
	'''
	def forward(self, embedding, out_planes=1, activation=None, use_bias=True):

		self.in_planes = embedding.data.shape[-1]
		self.out_planes = out_planes
		self.use_bias = use_bias

		self.linear = nn.Linear(self.in_planes, self.out_planes, bias=use_bias).cuda()
		if activation == 'relu':
			self.activation = nn.ReLU(inplace=True).cuda()
		elif activation == 'tanh':
			self.activation = nn.Tanh().cuda()

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight)
				if self.use_bias:
					nn.init.constant(m.bias, 0.1)

		out = self.linear(embedding)
		if self.activation is not None:
			out = self.activation(out)

		return out

'''
testing code
Arguments:
	- gpu_id
'''
if __name__ == '__main__':
	torch.cuda.set_device(int(sys.argv[1]))

	net = DynamicFC()
	test = Variable(torch.Tensor(4,64).cuda())
	out = net(test,4)
	print 'output: ', out.data.shape
