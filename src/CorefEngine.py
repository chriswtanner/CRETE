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

		# manually-defined features (others are in Resolver.py)
		#32, 20, 2, 32, 0
		wdPresets = [32, 5, 2, 32, 0] # batchsize, num epochs, num layers, num filters, dropout
		num_runs = 3

		entity_resolution = Resolver(args, wdPresets, "doc") # doc or dir for WD or CD, respectively
		
		# True means use pronouns, False means do not
		# {none, shortest, one} for supplemental path info
		entity_ids, entity_preds, entity_golds = entity_resolution.resolve("events", "relations", False, True, num_runs) 

		'''
		# second try
		event_resolution = Resolver(args, wdPresets, entity_ids, entity_preds)
		event_ids, event_preds, event_golds = event_resolution.resolve("events", "shortest", False, num_runs)

		entity_resolution = Resolver(args, wdPresets, event_ids, event_preds)
		entity_ids, entity_preds, entity_golds = entity_resolution.resolve("entities", "shortest", True, num_runs) # True means use pronouns, False means do not
		'''