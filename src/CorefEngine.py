# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
from ECBHelper import ECBHelper
class CorefEngine:

	# TODO:
	# - start writing ECBHelper, which takes HDDCRPParser's mentions and looks at each one's extension in order to figure out which DirHalf.addHM() to call
	# - sinc ei'll eventually need to write a new Gold File (CoNLL format but ECB Gold),
	#    i'll need to make sure that i point each HDDCRP UID to each Mention that it creates -- since some lines point to multiple mentions
	# - write StanfordParser
	# - compute stats on how many entities and events in ECB Corpus
	# - how many of these does Stanford contain? (for Ent+Events)
	# - make a new Gold file (CoNLL format) which
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
		helper.createHDDCRPMention(hddcrp_mentions)
