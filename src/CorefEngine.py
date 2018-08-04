# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
import time
import pickle
import random
import sys
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
from FeatureHandler import FeatureHandler
from DataHandler import DataHandler
from FFNN import FFNN
from CCNN import CCNN
from LibSVM import LibSVM
from HDF5Reader import HDF5Reader
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
	#     entity information: (1) all Ent+Events; (2) Ent+Events and remove singletons
	#      (3) Ents (minus pronouns)+Events
	# measure performance on entities, events, entities+events:
	# - (1) how well does our system (CCNN+AGG) do on:
	#	 (A) test on all and hope our system doesn't mix ents and events
	#    (B) test on non-events
	#    (C) test on non-events non-pronouns
 	# - (2) how well does StanCoreNLP do on:
	#	 (A) test on all and hope our system doesn't mix ents and events
	#    (B) test on non-events

	if __name__ == "__main__":
		wordFeaturesFile = "../data/features/word.f"
		lemmaFeaturesFile = "../data/features/lemma.f"
		charFeaturesFile = "../data/features/char.f"
		posFeaturesFile = "../data/features/pos.f"
		dependencyFeaturesFile = "../data/features/dependency.f"
		wordnetFeaturesFile = "../data/features/wordnet.f"
		bowFeaturesFile = "../data/features/bow.f"

		runStanford = False
		
		# classifier params
		numRuns = 2
		useCCNN = True
		useRelationalFeatures = False
		wdPresets = [64, 5, 2, 32, 0.0]

		# handles passed-in args
		args = params.setCorefEngineParams()
		
		start_time = time.time()
		# parses elmo embeddings
		#elmo = HDF5Reader('/Users/christanner/research/CRETE/data/features/alloutput3.hdf5')
		#exit(1)
		# most functionality lives here
		helper = ECBHelper(args)

		# parses the real, actual corpus (ECB's XML files)
		ecb_parser = ECBParser(args, helper)
		corpus = ecb_parser.parseCorpus(helper.docToVerifiedSentences)

		helper.addECBCorpus(corpus)
		helper.printCorpusStats()

		# parses the HDDCRP Mentions
		if not args.useECBTest:
			hddcrp_parser = HDDCRPParser(args)
			helper.createHDDCRPMentions(hddcrp_parser.parseCorpus(args.hddcrpFullFile))
		
		#exit(1)
		# loads Stanford's parse
		if runStanford:
			stan = StanParser(args, corpus)
			helper.addStanfordAnnotations(stan)
			helper.saveStanTokens()
		else:
			helper.loadStanTokens()
		helper.createStanMentions()
		#helper.printHDDCRPMentionCoverage()
		corpus.checkMentions()

		# DEFINES WHICH MENTIONS TO USE
		trainMUIDs = set()
		devMUIDs = set()
		testMUIDs = set()
		for m in corpus.ecb_mentions:
			if m.dir_num in helper.trainingDirs:
				trainMUIDs.add(m.XUID)
			elif m.dir_num in helper.devDirs:
				devMUIDs.add(m.XUID)
			elif m.dir_num in helper.testingDirs:
				testMUIDs.add(m.XUID)
		#for m in corpus.hddcrp_mentions:
		#	testMUIDs.add(m.XUID)
		fh = FeatureHandler(args, helper, trainMUIDs, devMUIDs, testMUIDs)

		'''
		fh.saveLemmaFeatures(lemmaFeaturesFile)
		fh.saveCharFeatures(charFeaturesFile)
		fh.savePOSFeatures(posFeaturesFile)
		fh.saveDependencyFeatures(dependencyFeaturesFile)
		fh.saveWordNetFeatures(wordnetFeaturesFile)
		fh.saveBoWFeatures(bowFeaturesFile)
		'''
		dh = DataHandler(fh, helper)
		#model = LibSVM(helper, coref)

		# within-doc first, then cross-doc
		if useCCNN:
			wd_model = CCNN(helper, dh, useRelationalFeatures, True, wdPresets)
			wd_model.train_and_test_wd(1) # 1 means only 1 run of WD
			cd_model = CCNN(helper, dh, useRelationalFeatures, False, [])
		else:
			wd_model = FFNN(helper, dh)
			wd_model.train_and_test_wd(numRuns)
		
		print("* done.  took ", str((time.time() - start_time)), "seconds")
		exit(1) # Tensorflow takes a long time to close sessions, so let's just kill the program
