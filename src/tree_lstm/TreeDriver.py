import random
import os
import sys
tree_dir = os.getcwd()
if "tree_lstm" not in os.getcwd():
	os.chdir("tree_lstm/")
	tree_dir += "/tree_lstm/"
	if tree_dir not in sys.path:
		sys.path.insert(0, tree_dir)

import config
from Helper import Helper
import Constants
from Vocab import Vocab # DATA HANDLING CLASSES
from SICKDataset import SICKDataset # Tree format of the SICK corpus
from Metrics import Metrics # METRICS CLASS FOR EVALUATION
from SimilarityTreeLSTM import SimilarityTreeLSTM
from Trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
class TreeDriver():
	def __init__(self, scope, num_dirs, opt="adagrad", lr=0.025):
		print("num_dirs:", num_dirs)
		sub_dir = "ecb_" + scope + "/"

		# init stuff
		print("TORCH VERSION:", torch.__version__)
		global args
		self.args = config.parse_known_args()
		
		self.args.cuda = self.args.cuda and torch.cuda.is_available()
		device = torch.device("cuda:0" if self.args.cuda else "cpu")
		torch.manual_seed(self.args.seed)
		random.seed(self.args.seed)

		print("TREELSTM:", opt, "lr:", lr)

		# paths
		train_dir = os.path.join(self.args.data, str(num_dirs), 'train/', sub_dir)
		dev_dir = os.path.join(self.args.data, str(num_dirs), 'dev/', sub_dir)
		test_dir = os.path.join(self.args.data, str(num_dirs), 'test/', sub_dir)

		print("train_dir:", train_dir)
		print("dev_dir:", dev_dir)

		# builds vocabulary
		sick_vocab_file = Helper.build_entire_vocab(os.path.join(self.args.data, str(num_dirs), 'sick.vocab'), train_dir, dev_dir, test_dir)
		vocab = Vocab(filename=sick_vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
		print('==> SICK vocabulary size : %d ' % vocab.size())

		# loads SICKDataset: Trees, sentences, and labels
		self.train_dataset = Helper.load_data(train_dir, os.path.join(self.args.data, str(num_dirs), 'sick_train.pth'), vocab, self.args.num_classes)
		self.dev_dataset = Helper.load_data(dev_dir, os.path.join(self.args.data, str(num_dirs), 'sick_dev.pth'), vocab, self.args.num_classes)
		self.test_dataset = Helper.load_data(test_dir, os.path.join(self.args.data, str(num_dirs), 'sick_test.pth'), vocab, self.args.num_classes)

		# creates the TreeLSTM
		model = SimilarityTreeLSTM(vocab.size(), self.args.input_dim, self.args.mem_dim, self.args.hidden_dim, \
				self.args.num_classes, self.args.sparse, self.args.freeze_embed, vocab)
		criterion = nn.KLDivLoss() #nn.CrossEntropyLoss()

		# loads glove embeddings
		emb = Helper.load_embeddings(self.args, os.path.join(self.args.data, str(num_dirs), 'sick_embed.pth'), vocab, device)

		# sets up the model
		model.emb.weight.data.copy_(emb) # plug these into embedding matrix inside model
		model.to(device)
		criterion.to(device)
		opt = optim.Adagrad(filter(lambda p: p.requires_grad, \
					model.parameters()), lr=lr, weight_decay=self.args.wd)

		if opt == "adam":
			opt = optim.Adam(filter(lambda p: p.requires_grad, \
					model.parameters()), lr=lr)

		self.metrics = Metrics(self.args.num_classes)
		
		# create trainer object for training and testing
		self.trainer = Trainer(self.args, model, criterion, opt, device, vocab)

	def train(self, epoch_num, bs): # performs one epoch of training
		train_loss = self.trainer.train(self.train_dataset, bs)
		train_loss, train_pred = self.trainer.test(self.train_dataset)
		train_pearson = self.metrics.pearson(train_pred, self.train_dataset.labels)
		train_mse = self.metrics.mse(train_pred, self.train_dataset.labels)
		print('==> Epoch {}, Train ({} examples) \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch_num, len(self.train_dataset), train_loss, train_pearson, train_mse))

	def fetch_hidden_embeddings(self, datum):
		return self.trainer.fetch_tree_nodes(datum)

	def test_on_train(self):
		print("test_on_train():")
		exit()
		highest_test_f1 = 0
		test_loss, test_pred = self.trainer.test(self.train_dataset)
		test_pearson = self.metrics.pearson(test_pred, self.train_dataset.labels)
		test_mse = self.metrics.mse(test_pred, self.train_dataset.labels)
		(test_f1, test_prec, test_rec, test_bestThreshold) = Helper.calculate_f1(test_pred, self.train_dataset.labels, True, True)
		print('==> Test \tLoss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(test_loss, test_pearson, test_mse, test_f1))
		if test_f1 > highest_test_f1:
			highest_test_f1 = test_f1
		print("**** highest_test_f1:", highest_test_f1)

	def test(self):
		test_loss, test_pred = self.trainer.test(self.test_dataset)
		test_pearson = self.metrics.pearson(test_pred, self.test_dataset.labels)
		test_mse = self.metrics.mse(test_pred, self.test_dataset.labels)
		(test_f1, test_prec, test_rec, test_bestThreshold) = Helper.calculate_f1(test_pred, self.test_dataset.labels, True, True)
		print('==> Test \tLoss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(test_loss, test_pearson, test_mse, test_f1))
		return test_f1, test_bestThreshold

'''
def main(train_set=None, dev_set=None, test_set=None):
	print("**** IN TREEDRIVER MAIN()")
	
	best = -float('inf')
	highest_dev_f1 = 0
	highest_test_f1 = 0
	for epoch in range(args.epochs):
		train_loss = trainer.train(train_dataset)
		train_loss, train_pred = trainer.test(train_dataset)
		train_pearson = metrics.pearson(train_pred, train_dataset.labels)
		train_mse = metrics.mse(train_pred, train_dataset.labels)
		print('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, train_loss, train_pearson, train_mse))
		
		for datum in train_dataset:
			left_to_hidden, right_to_hidden = trainer.fetch_tree_nodes(datum)
			print("left_to_hidden:", left_to_hidden)
			print("datum:", datum)
			exit()

		dev_loss, dev_pred = trainer.test(dev_dataset)
		dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
		dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
		(dev_f1, dev_prec, dev_rec, dev_bestThreshold) = Helper.calculate_f1(dev_pred, dev_dataset.labels)
		print('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(epoch, dev_loss, dev_pearson, dev_mse, dev_f1))
		if dev_f1 > highest_dev_f1:
			highest_dev_f1 = dev_f1

		test_loss, test_pred = trainer.test(test_dataset)
		test_pearson = metrics.pearson(test_pred, test_dataset.labels)
		test_mse = metrics.mse(test_pred, test_dataset.labels)
		(test_f1, test_prec, test_rec, test_bestThreshold) = Helper.calculate_f1(test_pred, test_dataset.labels)
		print('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(epoch, test_loss, test_pearson, test_mse, test_f1))
		if test_f1 > highest_test_f1:
			highest_test_f1 = test_f1

		print("* done w/ 1 epoch")
		exit()
	print("highest_dev_f1:", highest_dev_f1)
	print("highest_test_f1:", highest_test_f1)
		
if __name__ == "__main__":
	print("we in __main__")
	main()
'''