# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params
from ECBParser import ECBParser
from HDDCRPParser import HDDCRPParser
class CorefEngine:

	# TODO:
	# X - adjust Mention class so that the initializer sets "dirHalf"
	# - write HDDCRPParser, which will reuse the Mention class but store them in DirHalf and Doc as HMs.  nested setters, just like I do for DMs
	# - 	and generates all Mentions merely as hddcrp_parser.mentions
	# - start writing ECBHelper, which takes HDDCRPParser's mentions and looks at each one's extension in order to figure out which DirHalf.addHM() to call
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

		# parses the HDDCRP Mentions
		hddcrp_parser = HDDCRPParser(args)
		hddcrp_corpus = hddcrp_parser.parseCorpus(args.hddcrpFullFile)

