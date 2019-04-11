from tqdm import tqdm
import torch
import Helper as helper

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
		print("# training examples:", len(dataset))
		for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
			#print("train: idx:", idx)
			ltree, lsent, lparents, rtree, rsent, rparents, label = dataset[idx]
			lwords = " ".join([self.vocab.idxToLabel[int(x)] for x in lsent])
			rwords = " ".join([self.vocab.idxToLabel[int(x)] for x in rsent])
			
			#print("ltree:", ltree)
			#self.dfs_tree(ltree, 0)
			#self.dfs_tree(rtree, 0)
			target = helper.map_label_to_target(label, dataset.num_classes)

			lsent, rsent = lsent.to(self.device), rsent.to(self.device)
			target = target.to(self.device)

			#, "; target:", target)
			calculate_sim = False
			if idx == 0:
				calculate_sim = True
			output = self.model(ltree, lsent, lparents, rtree, rsent, rparents, calculate_sim)
			
			exit()
			#print("idx:", idx, "\n\tlabel:", label, "\n\tlwords:", lwords, "\n\trinput:", rwords, "\n\toutput:", output, "\n\ttarget:", target)
			loss = self.criterion(output, target)
			if idx == 6:
				print("target:", target, "; output:", output, "; loss:", loss)

			#ltree, "output:", output, "target:", target)
			total_loss += loss.item()
			loss.backward()
			
			if idx % self.args.batchsize == 0 and idx > 0:
				self.optimizer.step()
				self.optimizer.zero_grad()
		
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
				
				target = helper.map_label_to_target(label, dataset.num_classes)
				lsent, rsent = lsent.to(self.device), rsent.to(self.device)
				target = target.to(self.device)
				output = self.model(ltree, lsent, lparents, rtree, rsent, rparents, False)
				loss = self.criterion(output, target)
				total_loss += loss.item()
				output = output.squeeze().to('cpu')
				predictions[idx] = torch.dot(indices, torch.exp(output))
		return total_loss / len(dataset), predictions
	
