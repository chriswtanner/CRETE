import torch
import torch.nn as nn
import torch.nn.functional as F

import Constants

import operator
import math
import numpy as np

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
	def __init__(self, in_dim, mem_dim, vocab):
		super(ChildSumTreeLSTM, self).__init__()
		print("[] ChildSumTreeLSTM init()")
		self.in_dim = in_dim
		self.mem_dim = mem_dim
		self.vocab = vocab
		self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
		self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
		self.fx = nn.Linear(self.in_dim, self.mem_dim)
		self.fh = nn.Linear(self.mem_dim, self.mem_dim)

	def node_forward(self, token_embedding, children_c, children_h):
		#print("\t* ChildSumTreeLSTM.node_forward(): received:", len(token_embedding))
		
		#print("\t* ChildSumTreeLSTM.node_forward(): received:", len(inputs), "child_c:", len(child_c), "child_h:", len(child_h))
		children_h_sum = torch.sum(children_h, dim=0, keepdim=True)
		#print("\tchild_h_sum:", len(child_h_sum))

		iou = self.ioux(token_embedding) + self.iouh(children_h_sum) # adds in-place (overwrites values.  same dim)
		i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
		i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

		f = F.sigmoid(
			self.fh(children_h) +
			self.fx(token_embedding).repeat(len(children_h), 1)
		)
		fc = torch.mul(f, children_c)

		c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
		h = torch.mul(o, F.tanh(c))
		#print("returning c:", len(c), "and h:", len(h))
		return c, h

	def forward(self, tree, token_embeddings, index_to_hidden):
		#print("\tChildSumTreeLSTM.forward(): receiving ROOT tree:", tree, "; and inputs:", inputs[0][0:5])
		#print("\t\t# children:", tree.num_children)
		for idx in range(tree.num_children):
			#print("idx:", idx, self.vocab.idxToLabel[idx])
			self.forward(tree.children[idx], token_embeddings, index_to_hidden)

		if tree.num_children == 0:
			#print("*** no children!")
			child_c = token_embeddings[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
			child_h = token_embeddings[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
		else:
			#print("* we have # children:", tree.num_children)
			child_c, child_h = zip(* map(lambda x: x.state, tree.children))
			#print("len(child_h)1:", len(child_h))
			child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
			#print("len(child_h)2:", len(child_h))
			#print("child_c:", child_c)
			#print("child_h:", child_h)

		tree.state = self.node_forward(token_embeddings[tree.idx], child_c, child_h)
		index_to_hidden[tree.idx] = tree.state[1] # TODO   0 = cell state 1 = hidden
		#print("tree has index_to_hidden:", len(index_to_hidden.keys()))
		#print("\ttree idx:", tree.idx, "; self.vocab.idxToLabel:", self.vocab.idxToLabel[tree.idx])#inputs[tree.idx]:", inputs[tree.idx])
		return tree.state

# module for distance-angle similarity
class Similarity(nn.Module):
	def __init__(self, mem_dim, hidden_dim, num_classes):
		super(Similarity, self).__init__()
		print("[] Similarity init()")
		self.mem_dim = mem_dim
		self.hidden_dim = hidden_dim
		self.num_classes = num_classes
		self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
		self.wp = nn.Linear(self.hidden_dim, self.num_classes)

	def forward(self, lvec, rvec):
		#print("[] Similarity.forward()")
		mult_dist = torch.mul(lvec, rvec)
		abs_dist = torch.abs(torch.add(lvec, -rvec))
		vec_dist = torch.cat((mult_dist, abs_dist), 1)

		out = F.sigmoid(self.wh(vec_dist))
		out = F.log_softmax(self.wp(out), dim=1)
		return out

# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
	def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze, vocab):
		super(SimilarityTreeLSTM, self).__init__()
		print("[] SimilarityTreeLSTM init()")
		self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
		if freeze:
			print("*** FREEZE EMBEDDINGS!!!")
			self.emb.weight.requires_grad = False
		self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim, vocab)
		self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
		self.vocab = vocab

	def forward(self, ltree, lsent, lparents, rtree, rsent, rparents, calculate_sim):
		#print("* in forward: ltree:", ltree, "; linputs:", linputs)
		l_input_embeddings = self.emb(lsent)
		r_input_embeddings = self.emb(rsent)
		#print("linputs", linputs)

		left_to_hidden = {}
		right_to_hidden = {}
		lstate, lhidden = self.childsumtreelstm(ltree, l_input_embeddings, left_to_hidden)
		rstate, rhidden = self.childsumtreelstm(rtree, r_input_embeddings, right_to_hidden)

		if calculate_sim:
			self.calc_sim(lsent, lparents, left_to_hidden, rsent, rparents, right_to_hidden)
		output = self.similarity(lhidden, rhidden) # TODO: change

		'''
		print(len(lsent), " vs:", len(left_to_hidden.keys()))
		print("larr:", len(larr), "larr:", larr)
		if len(lsent) != len(left_to_hidden.keys()):
			print("* ERROR< diff sizes!")
			print("lsent:", lsent)
			print("left keys:", left_to_hidden.keys())
			print(len(lsent), " vs:", len(left_to_hidden.keys()))

			print("rsent:", len(rsent))
			print("right keys:", len(right_to_hidden.keys()))
		'''
		return output, left_to_hidden, right_to_hidden

	def calc_sim(self, lsent, lparents, left_to_hidden, rsent, rparents, right_to_hidden):
		sim_pairs = {}

		print("lsent:", lsent, "; lparents:", lparents, "; lroot:", lparents.index(0))

		#ltokens = " ".join([self.vocab.idxToLabel[t] for t in lsent])
		
		'''
		lpos = lparents.index(0)
		rpos = rparents.index(0)
		print(left_to_hidden.keys())
		print(right_to_hidden.keys())
		for l in left_to_hidden:
			for r in right_to_hidden:
				dp = 0
				denom1 = 0
				denom2 = 0
				v1 = left_to_hidden[l][0].detach().numpy()
				v2 = right_to_hidden[r][0].detach().numpy()
				for i in range(len(v1)):
					dp += v1[i] * v2[i]
					denom1 += v1[i]*v1[i]
					denom2 += v2[i]*v2[i]

				denom1 = math.sqrt(denom1)
				denom2 = math.sqrt(denom2)
				cs = -1
				if denom1 != 0 and denom2 != 0:
					cs = float(dp / (denom1 * denom2))
				sim_pairs[(l, r)] = cs
		sorted_x = sorted(
			sim_pairs.items(), key=operator.itemgetter(1), reverse=True)
		for i in sorted_x:
			l, r = i[0]
			print(self.vocab.idxToLabel[lsent.detach().numpy()[l]], "-", \
				self.vocab.idxToLabel[rsent.detach().numpy()[r]], ": ", i[1])
		'''