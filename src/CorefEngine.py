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
from Inference import Inference
from FFNN import FFNN
from LibSVM import LibSVM
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

		from sklearn.svm import LinearSVC
		from sklearn.datasets import make_classification
		X = [[0, 0], [1, 1]]
		y = [0, 1]
		clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='auto',
                    kernel='linear', max_iter=-1, probability=False, random_state=None,
                    shrinking=True, tol=0.001, verbose=False)
		clf.fit(X, y)
		print(clf.predict([[2., 2.]]))
		'''
		SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, \
		decision_function_shape='ovr', degree=3, gamma='auto', \
		kernel='rbf', max_iter=-1, probability=False, random_state=None, \
		shrinking=True, tol=0.001, verbose=False)
		
		'''
		#exit(1)
		'''
		#X, y = make_classification(n_features=4, random_state=0)
		#clf = LinearSVC(random_state=0)
		#clf.fit(X, y)
		# [ [example1], [example2] ]
		# y should be [ scalars ]
		LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=0)
		#print(clf.coef_)
		#print(clf.intercept_)
		#print(clf.predict([[0, 0, 0, 0],[1,1,1,1]]))
		'''
		wordFeaturesFile = "../data/features/word.f"
		lemmaFeaturesFile = "../data/features/lemma.f"
		charFeaturesFile = "../data/features/char.f"
		posFeaturesFile = "../data/features/pos.f"
		dependencyFeaturesFile = "../data/features/dependency.f"
		wordnetFeaturesFile = "../data/features/wordnet.f"
		bowFeaturesFile = "../data/features/bow.f"

		runStanford = False

		# classifier params
		numRuns = 1
		useWD = True
		useRelationalFeatures = True

		start_time = time.time()

		# handles passed-in args
		args = params.setCorefEngineParams()

		# most functionality lives here
		helper = ECBHelper(args)

		# parses the real, actual corpus (ECB's XML files)
		ecb_parser = ECBParser(args, helper)
		corpus = ecb_parser.parseCorpus(helper.docToVerifiedSentences)

		helper.addECBCorpus(corpus)
		helper.printCorpusStats()
		# parses the HDDCRP Mentions
		hddcrp_parser = HDDCRPParser(args)
		helper.createHDDCRPMentions(hddcrp_parser.parseCorpus(args.hddcrpFullFile))

		# loads Stanford's parse
		
		if runStanford:
			stan = StanParser(args, corpus)
			helper.addStanfordAnnotations(stan)
			helper.saveStanTokens()
		else:
			helper.loadStanTokens()
		helper.createStanMentions()
		

		#helper.printHDDCRPMentionCoverage()
		#corpus.checkMentions()

		# DEFINES WHICH MENTIONS TO USE
		trainMUIDs = set()
		devMUIDs = set()
		testMUIDs = set()
		for m in corpus.ecb_mentions:
			if m.dir_num in helper.trainingDirs:
				trainMUIDs.add(m.XUID)
			elif m.dir_num in helper.devDirs:
				devMUIDs.add(m.XUID)
		for m in corpus.hddcrp_mentions:
			testMUIDs.add(m.XUID)

		fh = FeatureHandler(args, helper, trainMUIDs, devMUIDs, testMUIDs)
		
		'''
		fh.saveWordFeatures(wordFeaturesFile)
		fh.saveLemmaFeatures(lemmaFeaturesFile)
		fh.saveCharFeatures(charFeaturesFile)
		fh.savePOSFeatures(posFeaturesFile)
		fh.saveDependencyFeatures(dependencyFeaturesFile)
		fh.saveWordNetFeatures(wordnetFeaturesFile)
		fh.saveBoWFeatures(bowFeaturesFile)
		'''

		coref = Inference(fh, helper, useRelationalFeatures, useWD)
		model = LibSVM(helper, coref)
		#model = FFNN(helper, coref)
		preds = model.train_and_test(numRuns)

		print("took:", str((time.time() - start_time)), "seconds")
