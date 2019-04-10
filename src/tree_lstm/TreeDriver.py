import random
import os
import torch
import torch.nn as nn
import Helper as helper

import Constants

import torch.optim as optim

from config import parse_args # CONFIG PARSER
from Vocab import Vocab # DATA HANDLING CLASSES
from SICKDataset import SICKDataset # Tree format of the SICK corpus
from Metrics import Metrics # METRICS CLASS FOR EVALUATION
from SimilarityTreeLSTM import SimilarityTreeLSTM
from Trainer import Trainer
def main():

	# init stuff
	global args
	args = parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()
	device = torch.device("cuda:0" if args.cuda else "cpu")
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	# paths
	train_dir = os.path.join(args.data, 'train/', args.sub_dir)
	dev_dir = os.path.join(args.data, 'dev/', args.sub_dir)
	test_dir = os.path.join(args.data, 'test/', args.sub_dir)

	# builds vocabulary
	sick_vocab_file = helper.build_entire_vocab(os.path.join(args.data, 'sick.vocab'), train_dir, dev_dir, test_dir)
	vocab = Vocab(filename=sick_vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
	print('==> SICK vocabulary size : %d ' % vocab.size())
	print("vocab.idxToLabel:", vocab.idxToLabel)

	# loads Trees, sentences, and labels
	train_dataset = helper.load_data(train_dir, os.path.join(args.data, 'sick_train.pth'), vocab, args.num_classes)
	dev_dataset = helper.load_data(dev_dir, os.path.join(args.data, 'sick_dev.pth'), vocab, args.num_classes)
	test_dataset = helper.load_data(test_dir, os.path.join(args.data, 'sick_test.pth'), vocab, args.num_classes)

	# creates the TreeLSTM
	model = SimilarityTreeLSTM(vocab.size(), args.input_dim, args.mem_dim, args.hidden_dim, \
			args.num_classes, args.sparse, args.freeze_embed, vocab)
	criterion = nn.KLDivLoss()

	# loads glove embeddings
	emb = helper.load_embeddings(args, os.path.join(args.data, 'sick_embed.pth'), vocab, device)

	# sets up the model
	model.emb.weight.data.copy_(emb) # plug these into embedding matrix inside model
	model.to(device)
	criterion.to(device)
	optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, \
				model.parameters()), lr=args.lr, weight_decay=args.wd)

	metrics = Metrics(args.num_classes)
	
	# create trainer object for training and testing
	trainer = Trainer(args, model, criterion, optimizer, device, vocab)

	best = -float('inf')
	for epoch in range(args.epochs):
		train_loss = trainer.train(train_dataset)

		train_loss, train_pred = trainer.test(train_dataset)
		train_pearson = metrics.pearson(train_pred, train_dataset.labels)
		train_mse = metrics.mse(train_pred, train_dataset.labels)
		print('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, train_loss, train_pearson, train_mse))
			
		dev_loss, dev_pred = trainer.test(dev_dataset)
		dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
		dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
		print('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, dev_loss, dev_pearson, dev_mse))

		test_loss, test_pred = trainer.test(test_dataset)
		test_pearson = metrics.pearson(test_pred, test_dataset.labels)
		test_mse = metrics.mse(test_pred, test_dataset.labels)
		print('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch, test_loss, test_pearson, test_mse))
		
if __name__ == "__main__":
	main()