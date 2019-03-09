import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import dgl.function as fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgl.data.tree import SST
from dgl.data import SSTBatch

# Each sample in the dataset is a constituency tree. The leaf nodes
# represent words. The word is a int value stored in the "x" field.
# The non-leaf nodes has a special word PAD_WORD. The sentiment
# label is stored in the "y" feature field.
def batcher(dev):
	def batcher_dev(batch):
		batch_trees = dgl.batch(batch)
		return SSTBatch(graph=batch_trees,
			mask=batch_trees.ndata['mask'].to(device),
			wordid=batch_trees.ndata['x'].to(device),
			label=batch_trees.ndata['y'].to(device))
	return batcher_dev
	
trainset = SST(mode='tiny')  # the "tiny" set has only 5 trees
print("trainset:", trainset)
tiny_sst = trainset.trees
num_vocabs = trainset.num_vocabs
num_classes = trainset.num_classes
vocab = trainset.vocab # vocabulary dict: key -> id
inv_vocab = {v: k for k, v in vocab.items()} # inverted vocabulary dict: id -> word

a_tree = tiny_sst[0]
#print("tiny_sst:", tiny_sst)
tiny_sst = [tiny_sst[4]]

graph = dgl.batch(tiny_sst)
def plot_tree(g):
	# this plot requires pygraphviz package
	pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
	nx.draw(g, pos, with_labels=False, node_size=10,
		node_color=[[.5, .5, .5]], arrowsize=4)
	plt.show()

plot_tree(graph.to_networkx())

class TreeLSTM(nn.Module):
	def __init__(self,
		num_vocabs,
		x_size,
		h_size,
		num_classes,
		dropout,
		pretrained_emb=None):
		self.num_nodes = 0
		super(TreeLSTM, self).__init__()
		self.x_size = x_size
		self.embedding = nn.Embedding(num_vocabs, x_size)
		if pretrained_emb is not None:
			print('Using glove')
			self.embedding.weight.data.copy_(pretrained_emb)
			self.embedding.weight.requires_grad = True
		self.dropout = nn.Dropout(dropout)
		self.linear = nn.Linear(h_size, num_classes)
		self.cell = TreeLSTMCell(x_size, h_size)

	def forward(self, batch, h, c):
		"""Compute tree-lstm prediction given a batch.

		Parameters
		----------
		batch : dgl.data.SSTBatch
			The data batch.
		h : Tensor
			Initial hidden state.
		c : Tensor
			Initial cell state.

		Returns
		-------
		logits : Tensor
			The prediction of each node.
		"""
		g = batch.graph
		g.register_message_func(self.cell.message_func)
		g.register_reduce_func(self.cell.reduce_func)
		g.register_apply_node_func(self.cell.apply_node_func)
		# feed embedding
		embeds = self.embedding(batch.wordid * batch.mask)
		g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
		g.ndata['h'] = h
		g.ndata['c'] = c
		# propagate
		dgl.prop_nodes_topo(g)
		# compute logits
		h = self.dropout(g.ndata.pop('h'))
		print("h:", len(h))
		logits = self.linear(h)
		return logits

class TreeLSTMCell(nn.Module):
	def __init__(self, x_size, h_size):
		super(TreeLSTMCell, self).__init__()
		self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
		self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
		self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
		self.U_f = nn.Linear(2 * h_size, 2 * h_size)


	def message_func(self, edges):
		return {'h': edges.src['h'], 'c': edges.src['c']}

	def reduce_func(self, nodes):
		#print("in reduce_func()")
		# concatenate h_jl for equation (1), (2), (3), (4)
		h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
		# equation (2)
		f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
		# second term of equation (5)
		c = th.sum(f * nodes.mailbox['c'], 1)
		return {'iou': self.U_iou(h_cat), 'c': c}

	def apply_node_func(self, nodes):

		# equation (1), (3), (4)
		iou = nodes.data['iou'] + self.b_iou
		i, o, u = th.chunk(iou, 3, 1)
		i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
		# equation (5)
		c = i * u + nodes.data['c']
		# equation (6)
		h = o * th.tanh(c)
		
		return {'h' : h, 'c' : c}

device = th.device('cpu')
# hyper parameters
x_size = 256
h_size = 256
dropout = 0.5
lr = 0.05
weight_decay = 1e-4
epochs = 10

# create the model
model = TreeLSTM(trainset.num_vocabs, x_size, h_size, trainset.num_classes, dropout)
print(model)

# create the optimizer
optimizer = th.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

train_loader = DataLoader(dataset=tiny_sst,
	batch_size=10,
	collate_fn=batcher(device),
	shuffle=False,
	num_workers=0)

print("train_loader:", train_loader)

# training loop
for epoch in range(epochs):
	for step, batch in enumerate(train_loader):
		g = batch.graph
		n = g.number_of_nodes()
		h = th.zeros((n, h_size))
		c = th.zeros((n, h_size))
		logits = model(batch, h, c)
		logp = F.log_softmax(logits, 1)
		loss = F.nll_loss(logp, batch.label, reduction='sum')
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		pred = th.argmax(logits, 1)
		acc = float(th.sum(th.eq(batch.label, pred))) / len(batch.label)
		print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
			epoch, step, loss.item(), acc))