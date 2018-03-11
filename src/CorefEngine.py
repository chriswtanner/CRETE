# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
from StanParser import StanParser
class CorefEngine:

	# TODO:
	# - write StanfordParser
	# - compute stats on how many entities and events in ECB Corpus -- TRAIN/DEV/TEST
	# - how many of these does Stanford contain? (for Ent+Events) -- TRAIN/DEV
	#	- vary the cutoff point (N=inf, 10,9,8,7,6,5,4,3,2,1) -- TRAIN/DEV
	# - make 2 new Gold files (CoNLL format) which
	#   includes entity information: (1) all Ent+Events; (2) remove singletons

	if __name__ == "__main__":

		# handles passed-in args
		args = params.setCorefEngineParams()

		# parses the real, actual corpus (ECB's XML files)
		ecb_parser = ECBParser(args)
		ecb_corpus = ecb_parser.parseCorpus(args.corpusPath, args.verbose)

		# most functionality lives here
		helper = ECBHelper(args, ecb_corpus)
		
		# parses the HDDCRP Mentions
		hddcrp_parser = HDDCRPParser(args)
		hddcrp_mentions = hddcrp_parser.parseCorpus(args.hddcrpFullFile)
		helper.createHDDCRPMentions(hddcrp_mentions)

		# loads Stanford's parse
		stan = StanParser(args, ecb_corpus)
