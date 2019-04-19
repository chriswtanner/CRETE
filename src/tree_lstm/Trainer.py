from tqdm import tqdm
import torch

from Helper import Helper

class Trainer(object):
	def __init__(self, args, model, criterion, optimizer, device, vocab):
		super(Trainer, self).__init__()
		self.args = args
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.device = device
		self.epoch = 0

		self.vocab = vocab # I ADDED THIS

	def dfs_tree(self, node, depth):
		prefix = ""
		for i in range(depth):
			prefix += "\t"
		print(prefix, node.idx)
		for child in node.children:
			self.dfs_tree(child, depth+1)

	# helper function for training
	def train(self, dataset):
		self.model.train()
		self.optimizer.zero_grad()
		total_loss = 0.0
		#indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

		num_mismatched_dependencies = 0
		for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
			#print("train: idx:", idx)
			ltree, lsent, lparents, rtree, rsent, rparents, label = dataset[idx]
			lwords = " ".join([self.vocab.idxToLabel[int(x)] for x in lsent])
			rwords = " ".join([self.vocab.idxToLabel[int(x)] for x in rsent])
			
			target = Helper.map_label_to_target(label, dataset.num_classes)

			lsent, rsent = lsent.to(self.device), rsent.to(self.device)
			target = target.to(self.device)

			#, "; target:", target)
			calculate_sim = False
			if idx == -1:
				calculate_sim = True
			
			#print("\tlenlsent:", len(lsent), "lparents:", len(lparents))
			output, left_to_hidden, right_to_hidden = self.model(ltree, lsent, lparents, rtree, rsent, rparents, calculate_sim)
			#print("idx:", idx, "\n\tlabel:", label, "\n\tlwords:", lwords, "\n\trinput:", rwords, "\n\toutput:", output, "\n\ttarget:", target)
			loss = self.criterion(output, target)

			if len(left_to_hidden) != len(lsent):
				#print("TRAIN idx:", idx, "lwords:", lwords)
				#print("* ERROR: ", len(left_to_hidden), "!=", len(lsent))
				num_mismatched_dependencies += 1
				
			#ltree, "output:", output, "target:", target)
			total_loss += loss.item()
			loss.backward()
			
			if idx % self.args.batchsize == 0 and idx > 0:
				self.optimizer.step()
				self.optimizer.zero_grad()
		
		print("*** num_mismatched_dependencies:", num_mismatched_dependencies)
		self.epoch += 1
		return total_loss / len(dataset)

	# helper function for testing
	def test(self, dataset):
		self.model.eval()
		with torch.no_grad():
			total_loss = 0.0
			predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
			indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
			for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
				ltree, lsent, lparents, rtree, rsent, rparents, label = dataset[idx]
				target = Helper.map_label_to_target(label, dataset.num_classes)
				lsent, rsent = lsent.to(self.device), rsent.to(self.device)
				target = target.to(self.device)
				output, _, _ = self.model(ltree, lsent, lparents, rtree, rsent, rparents, False)
				loss = self.criterion(output, target)
				total_loss += loss.item()
				output = output.squeeze().to('cpu')
				predictions[idx] = torch.dot(indices, torch.exp(output))
				if idx < -1:
					print("TEST idx:", idx, "; label:", label, "; target:", target, "; output:", output, "; preds:", predictions[idx])

		#print("DONE WITH Trainer.test()")
		return total_loss / len(dataset), predictions

	# passes in just 1 line of the SICK Dataset (2 trees)
	def fetch_tree_nodes(self, datum):
		self.model.eval()
		with torch.no_grad():
			ltree, lsent, lparents, rtree, rsent, rparents, label = datum
			lsent, rsent = lsent.to(self.device), rsent.to(self.device)
			output, left_to_hidden, right_to_hidden = self.model(ltree, lsent, lparents, rtree, rsent, rparents, False)
			lwords = " ".join([self.vocab.idxToLabel[int(x)] for x in lsent])
			rwords = " ".join([self.vocab.idxToLabel[int(x)] for x in rsent])
			#print("\ttree's received lwords:", lwords)
			#print("\ttree's received rwords:", rwords)
		return lwords, left_to_hidden, rwords, right_to_hidden
	