# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
import time
import pickle
from nltk.corpus import wordnet as wn
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
from FeatureHandler import FeatureHandler
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
		'''
		dog = wn.synsets('asj0zj0')
		print(dog)
		cat = wn.synsets('cat')[0]
		print(dog.res_similarity(cat))
		print(cat.wup_similarity(dog))

		exit(1)
		'''
		runStanford = False
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

		# DEFINE WHICH MENTIONS TO USE
		mentions = set()
		for m in corpus.ecb_mentions:
			if m.dir_num in helper.testingDirs:
				mentions.add(m)
		for m in corpus.hddcrp_mentions:
			mentions.add(m)

		fh = FeatureHandler(args, helper, mentions)
		#fh.saveWordNetFeatures("../data/features/wordnet.f")
		fh.saveBoWFeatures("../data/features/bow.f")
		print("took:", str((time.time() - start_time)), "seconds")
