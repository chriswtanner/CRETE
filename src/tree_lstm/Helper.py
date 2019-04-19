import os
import math
import torch
import sys
import numpy as np
print("helper sys path:", sys.path)
print("in helper; cwd:", os.getcwd())
import Constants
from Vocab import Vocab
from SICKDataset import SICKDataset
from collections import defaultdict
# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
class Helper:
	def load_word_vectors(path):
		if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
			print('==> File found, loading to memory')
			vectors = torch.load(path + '.pth')
			vocab = Vocab(filename=path + '.vocab')
			return vocab, vectors
		# saved file not found, read from txt file
		# and create tensors for word vectors
		print('==> File not found, preparing, be patient')
		count = sum(1 for line in open(path + '.txt', 'r', encoding='utf8', errors='ignore'))
		with open(path + '.txt', 'r') as f:
			contents = f.readline().rstrip('\n').split(' ')
			dim = len(contents[1:])
		words = [None] * (count)
		vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')
		with open(path + '.txt', 'r', encoding='utf8', errors='ignore') as f:
			idx = 0
			for line in f:
				contents = line.rstrip('\n').split(' ')
				words[idx] = contents[0]
				values = list(map(float, contents[1:]))
				vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
				idx += 1
		with open(path + '.vocab', 'w', encoding='utf8', errors='ignore') as f:
			for word in words:
				f.write(word + '\n')
		vocab = Vocab(filename=path + '.vocab')
		torch.save(vectors, path + '.pth')
		return vocab, vectors

	# uses dictionary of embeddings:
	# type_id -> [embedding]
	def load_embeddings(args, emb_file, sick_vocab, device):
		# if word has a Glove pre-trained embeddings, use it; otherwise, use random vector
		if False and os.path.isfile(emb_file):
			print("* load_embeddings() -- load the dict of embeddings")
			emb = torch.load(emb_file)

		else: # load glove embeddings and vocab
			print("* load_embeddings() -- creating dict of embeddings")
			glove_vocab, glove_emb = Helper.load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
			print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
			emb = torch.zeros(sick_vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
			emb.normal_(0, 0.05) # randomly initialize every word (then replace found ones w/ GLoVE)
			
			# zero out the embeddings for padding and other special words if they are absent in vocab
			for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
				emb[idx].zero_()

			for word in sick_vocab.labelToIdx.keys():
				if glove_vocab.getIndex(word):
					emb[sick_vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
			print("* saving emb (", len(emb), ") items to:", emb_file)
			torch.save(emb, emb_file)
		
		return emb

	def build_entire_vocab(sick_vocab_file, train_dir, dev_dir, test_dir):
		if True or not os.path.isfile(sick_vocab_file):
			print("need to build a SICK vocab:", sick_vocab_file)
			token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
			token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
			token_files = token_files_a + token_files_b

			vocab = set()
			for filename in token_files:
				with open(filename, 'r') as f:
					for line in f:
						tokens = line.rstrip('\n').split(' ')
						vocab |= set(tokens)
			with open(sick_vocab_file, 'w') as f:
				for token in sorted(vocab):
					f.write(token + '\n')
		else:
			print("* vocab already built!")
		return sick_vocab_file

	# write unique words from a set of files to a new file
	def build_vocab(filenames, vocabfile):
		print("* helper's build_vocab()")
		vocab = set()
		for filename in filenames:
			with open(filename, 'r') as f:
				for line in f:
					tokens = line.rstrip('\n').split(' ')
					vocab |= set(tokens)
		with open(vocabfile, 'w') as f:
			for token in sorted(vocab):
				f.write(token + '\n')

	def load_data(base_dir, file, vocab, num_classes):
		print("* helper's load_data(); file:", file)
		if False and os.path.isfile(file):
			print("\t* loaded it")
			dataset = torch.load(file)
		else:
			print("\t* we dont have it")
			dataset = SICKDataset(base_dir, vocab, num_classes)
			torch.save(dataset, file)

		print('\t==> Size of data split: %d ' % len(dataset))
		return dataset

	# mapping from scalar to vector
	def map_label_to_target(label, num_classes):
		'''
		target = torch.zeros(1, dtype=torch.long, device='cpu')
		target[0] = label-1
		#print(target)
		return target
		'''
		target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
		#print("label:", label, "; target:", target)
		ceil = int(math.ceil(label))
		floor = int(math.floor(label))
		if ceil == floor:
			target[0, floor-1] = 1
		else:
			target[0, floor-1] = ceil - label
			target[0, ceil-1] = label - floor
		#print("\ttarget:", target)
		#label = label -1
		#print(label)
		#return torch.tensor([label])
		return target
		
	def plot_distance_matrix(xuid_pairs, preds, golds, threshold, xuid_to_dist):
		key_to_counts = defaultdict(int)
		key_to_perf = defaultdict(lambda: defaultdict(int))
		max_dist = 0
		xuids_missing = set()
		for i, (xuid1, xuid2) in enumerate(xuid_pairs):
			predicted = False
			if preds[i] >= threshold:
				predicted = True
			is_gold = False
			if golds[i] == 2:
				is_gold = True
			
			# make dist1 the lower one
			missing = False
			if xuid1 not in xuid_to_dist:
				xuids_missing.add(xuid1)
				missing = True
			if xuid2 not in xuid_to_dist:
				xuids_missing.add(xuid2)
				missing = True
			if missing:
				continue
			dist1 = xuid_to_dist[xuid1]
			dist2 = xuid_to_dist[xuid2]
			if dist2 < dist1:
				tmp = dist1
				dist1 = dist2
				dist2 = tmp
			if dist2 > max_dist:
				max_dist = dist2

			key = str(dist1) + "_" + str(dist2)
			key_to_counts[key] += 1
			if predicted and is_gold: # TRUE POSITIVE
				key_to_perf[key]["TP"] += 1
			elif predicted and not is_gold: # FALSE POSITIVE
				key_to_perf[key]["FP"] += 1
			elif not predicted and is_gold:
				key_to_perf[key]["FN"] += 1
			elif not predicted and not is_gold:
				key_to_perf[key]["TN"] += 1
			
		print("# xuids_missing (for plotting):", xuids_missing)
	
		matrix_total = np.zeros(shape=(max_dist, max_dist))
		matrix_f1 = np.zeros(shape=(max_dist, max_dist))
		for key in key_to_counts:
			dist1, dist2 = key.split("_")
			dist1 = int(dist1) - 1
			dist2 = int(dist2) - 1
			recall = 0
			if key_to_perf[key]["TP"] > 0 or key_to_perf[key]["FN"] > 0:
				recall = key_to_perf[key]["TP"] / (key_to_perf[key]["TP"] + key_to_perf[key]["FN"])
			prec = 0
			if key_to_perf[key]["TP"] >0 or key_to_perf[key]["FP"] > 0:
				prec = key_to_perf[key]["TP"] / (key_to_perf[key]["TP"] + key_to_perf[key]["FP"])
			f1 = -1
			if recall != 0 and prec != 0:
				f1 = 2*recall*prec / (recall + prec)
			matrix_total[dist1, dist2] = key_to_counts[key]
			matrix_f1[dist1, dist2] = round(f1, 4)

		print("matrix_total:\n", matrix_total)
		print("matrix_f1\n:", matrix_f1)


	def calculate_f1(preds, golds, convert_from_tensor=False, rev=True):

		numGoldPos = 0
		scoreToGoldTruth = defaultdict(list)
		#print("golds:", golds)
		for _ in range(len(preds)):
			pred = preds[_] #.item()
			if golds[_] == 2:
				numGoldPos += 1
				scoreToGoldTruth[pred].append(1)
			else:
				scoreToGoldTruth[pred].append(0)
			#print("pred:", pred, "gold:", golds[_])

		s = sorted(scoreToGoldTruth.keys(), reverse=rev)
		TP = 0.0
		FP = 0.0
		bestF1 = 0
		bestVal = -1
		bestR = 0
		bestP = 0
		numReturnedSoFar = 0
		score_to_index_rank = {}
		for eachVal in s:
			for _ in scoreToGoldTruth[eachVal]:
				if _ == 1:
					TP += 1
				else:
					FP += 1

			numReturnedSoFar += len(scoreToGoldTruth[eachVal])

			if convert_from_tensor:
				eachVal = float(eachVal.numpy())
				
			score_rounded = str(round(eachVal,7))
			score_to_index_rank[score_rounded] = numReturnedSoFar

			recall = float(TP / numGoldPos)
			prec = float(TP / numReturnedSoFar)
			f1 = 0
			if (recall + prec) > 0:
				f1 = 2*(recall*prec) / (recall + prec)

			if f1 > bestF1:
				bestF1 = f1
				bestVal = eachVal
				bestR = recall
				bestP = prec
				#print("new besT:", bestF1, " (", bestVal, ")")
		if numReturnedSoFar != len(preds):
			print("* ERROR: we didn't look at preds correctly")
			exit(1)
		if bestF1 <0:
			print("* ERROR: our F1 was < 0")
			exit(1)
	
		return (bestF1, bestP, bestR, bestVal)