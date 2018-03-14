# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
import time
import pickle
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
class CorefEngine:

	# TODO:
	# - figure out correctness of ECB+ (does it annotate every sentence?)
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
	#	 (A) test on all and hope our system doens't mix ents and events
	#    (B) test on non-events
	if __name__ == "__main__":
		runStanford = False
		useEntities = False

		# handles passed-in args
		args = params.setCorefEngineParams()

		# parses the real, actual corpus (ECB's XML files)
		ecb_parser = ECBParser(args)
		corpus = ecb_parser.parseCorpus(args.corpusPath, args.verbose)
		
		# most functionality lives here
		helper = ECBHelper(args, corpus)
		helper.printCorpusStats()
		
		# parses the HDDCRP Mentions
		hddcrp_parser = HDDCRPParser(args)
		hddcrp_mentions = hddcrp_parser.parseCorpus(args.hddcrpFullFile)
		helper.createHDDCRPMentions(hddcrp_mentions)

		start_time = time.time()
		
		# loads Stanford's parse
		if runStanford:
			stan = StanParser(args, corpus)
			helper.addStanfordAnnotations(stan)
			helper.saveStanTokens()
		else:
			helper.loadStanTokens()
		print("took:", str((time.time() - start_time)), "seconds")
		
