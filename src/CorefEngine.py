# AUTHOR: Chris Tanner (christanner@cs.brown.edu) 
# PURPOSE: CRETE: Coreference Resolution for EnTities and Events (uses ECB+ corpus)
import params

class CorefEngine:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setCorefEngineParams()

		# parses the real, actual corpus (ECB's XML files)
		corpus = ECBParser(args)
		#helper = ECBHelper(args, corpus, hddcrp_parsed, runFFNN)

		#hddcrp_parsed = HDDCRPParser(args)

