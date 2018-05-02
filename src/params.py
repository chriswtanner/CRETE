import argparse

def setCorefEngineParams():
	parser = argparse.ArgumentParser()

	# ECBParser
	parser.add_argument("--corpusPath", help="the corpus dir")
	parser.add_argument("--useECBTest", help="use the ECB test set or not", type=str2bool, nargs='?', default="f")
	parser.add_argument("--onlyValidSentences", help="True or False, to use only the sentences that ECB+ claims to be good", type=str2bool, nargs='?')
	parser.add_argument("--addIntraDocs", help="True if we should add the intra-doc ones that have no cross-doc", type=str2bool, nargs='?')
	parser.add_argument("--stanTokensFile", help="pickled file containing StanTokens")
	parser.add_argument("--verifiedSentencesFile", help="file containing the ECB+ sentences that are trustworthy")
	parser.add_argument("--replacementsFile", help="we replace all instances of these tokens which appear in our corpus -- this is to help standardize the format, useful for creating embeddings and running stanfordCoreNLP")
	parser.add_argument("--verbose", help="print a lot of debugging info", type=str2bool, nargs='?', default="f")

	# StanParser
	parser.add_argument("--stanOutputDir", help="the file that stanfordCoreNLP output'ed")

	# CCNN
	parser.add_argument("--baseDir", help="the full path to /, where results/ and data/ reside")
	parser.add_argument("--hddcrpFullFile", help="the fullpath of HDDCRP File (gold or predict)")
	parser.add_argument("--numLayers", help="1 or 2 conv sections", type=int)
	parser.add_argument("--embeddingsFile", help="the full path of the embeddings file")
	parser.add_argument("--numNegPerPos", help="# of neg examples per pos in training (e.g., 1,2,5)", type=int)
	parser.add_argument("--numEpochs", help="type or token", type=int)
	parser.add_argument("--batchSize", help="batchSize", type=int)
	parser.add_argument("--windowSize", help="# of tokens before/after the Mention to use", type=int)
	parser.add_argument("--dropout", help="initial dropout rate", type=float)
	parser.add_argument("--numFilters", help="num CNN filters", type=int)
	parser.add_argument("--filterMultiplier", help="the \% of how many filters to use at each successive layer", type=float)
	# optionally added features to the CCNN
	# note, we used to read in the files for embeddings when we wanted to use
	# lemma, distinct from the regular word embeddings and dependency embeddings
	# but it makes most sense to just read in 1 set of embeddings (which have all the word types we'd care to use)
	parser.add_argument("--charEmbeddingsFile",
	                    help="the char embeddings file", default="none")
	parser.add_argument("--posEmbeddingsFile", help="the POS embeddings file", default="none")
	parser.add_argument("--wordFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--lemmaFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--charFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--posFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--dependencyFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--bowFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--wordnetFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	parser.add_argument("--framenetFeature", help="print a lot of debugging info",
	                    type=str2bool, nargs='?', default="f")
	# for FFNN
	parser.add_argument("--devDir", help="the directory to use for dev", type=int)
	parser.add_argument("--FFNNnumEpochs", help="FFNN's # of epochs", default="none", type=int)

	parser.add_argument("--native", help="true if running locally",type=str2bool, nargs='?', default="f")
	return parser.parse_args()

# allows for handling boolean params
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
