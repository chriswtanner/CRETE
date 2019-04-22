# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
import time
import pickle
import random
import sys
import os
import fnmatch
import numpy as np

from collections import defaultdict
from KBPParser import KBPParser
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
from FeatureHandler import FeatureHandler
from DataHandler import DataHandler
from FFNN import FFNN
from CCNN import CCNN
from Resolver import Resolver
#from LibSVM import LibSVM
#from HDF5Reader import HDF5Reader
from sklearn import svm


import keras
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import losses
from math import sqrt, floor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from collections import defaultdict
import torch
#from tree_lstm import TreeDriver
class CorefEngine:

	# TODO:
	# X - Q2: which ones do HDDCRP include?
	# Q3B: which ones do Stanford include? (NER)
	# Q3A: which ones do Stanford include? (coref only)
	# Q3C: which ones do Stanford include? (coref only+NER)
	# - how many of these does Stanford contain? (for Ent+Events) -- TRAIN/DEV
	#	- vary the cutoff point (N=inf, 10,9,8,7,6,5,4,3,2,1) -- TRAIN/DEV
	# - make 2 new Gold files (CoNLL format) which includes
	#     entity information: (1) all Ent+Events; (2) Ent+Events and remove sindgletons
	#      (3) Ents (minus pronouns)+Events
	if __name__ == "__main__":
		# handles passed-in args
		args = params.setCorefEngineParams()

		#saved_treelstm = pickle.load(open("doc_21_e9.p", "rb"))
		#print(saved_treelstm.xuid_pairs_to_index)
		#print(saved_treelstm.xuid_pairs_to_index['3905_3907'])
		#exit()
		# manually-defined features (others are in Resolver.py)
		#32, 20, 2, 32, 0
		wdPresets = [256, 2, 2, 4, 0] # batchsize, num epochs, num layers, num filters, dropout
		num_runs = 1
		mention_types = {'events'} #, 'entities'} # NOTE: should be 'events' and/or 'entities'

		event_resolution = Resolver(args, wdPresets, "dir") # doc or dir for WD or CD, respectively
		
		# {none, relations, shortest, one} for supplemental path info
		# resolve(mention_type, supp_features_type, event_pronouns, entity_pronouns, num_runs)
		# supp_features_type could be {none, shortest, one, type}
		event_ids, event_preds, event_golds = event_resolution.resolve(mention_types, "shortest", True, True, num_runs)

		#event_resolution.aggCluster(event_ids, event_preds, event_golds)

		'''
		# second try
		event_resolution = Resolver(args, wdPresets, entity_ids, entity_preds)
		event_ids, event_preds, event_golds = event_resolution.resolve("events", "shortest", False, num_runs)

		entity_resolution = Resolver(args, wdPresets, event_ids, event_preds)
		entity_ids, entity_preds, entity_golds = entity_resolution.resolve("entities", "shortest", True, num_runs) # True means use pronouns, False means do not
		'''