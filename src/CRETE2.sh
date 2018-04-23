#!/bin/bash
export PYTHONIOENCODING=UTF-8

# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/CRETE/"
brownDir="/home/ctanner/researchcode/CRETE/"

if [ ${me} = "ctanner" ]
then
	echo "[ RUNNING ON BROWN NETWORK ]"
	baseDir=${brownDir}
	refDir=${refDirBrown}
	if [ ${hn} = "titanx" ]
	then
		export CUDA_HOME=/usr/local/cuda/
	else
		source ~/researchcode/venv/bin/activate
		export CUDA_HOME=/contrib/projects/cuda8.0
	fi
	export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
	export PATH=${CUDA_HOME}/bin:${PATH}
else
	echo "[ RUNNING LOCALLY ]"
fi

hddcrpFullFile=${baseDir}"data/predict.ran.WD.semeval.txt" # MAKE SURE THIS IS WHAT YOU WANT (gold vs predict)
verbose="true"
embeddingsFile=${baseDir}"data/gloveEmbeddings.6B.300.txt"
scriptDir=${baseDir}"src/"
dataDir=${baseDir}"data/"
resultsDir=${baseDir}"results/"
refDir=${scriptDir}"reference-coreference-scorers-8.01/"
replacementsFile=${baseDir}"data/replacements.txt"
stanTokensFile=${baseDir}"data/stan_tokens.p"
verifiedSentencesFile=${baseDir}"data/ECBplus_coreference_sentences.csv"

charEmbeddingsFile=${baseDir}"data/charRandomEmbeddings.txt"

stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"

corpusPath=${baseDir}"data/ECB_$1/"
useECBTest=$2
onlyValidSentences=$3
numLayers=$4
numEpochs=$5
windowSize=$6
numNegPerPos=$7
batchSize=$8

# CCNN features 
dropout=${9}
numFilters=${10}
filterMultiplier=${11}
posType=${12}
posEmbeddingsFile=${baseDir}"data/posEmbeddings100.txt"
lemmaType=${13}
dependencyType=${14}
charType=${15}
devDir=${16}
FFNNnumEpochs=${17}
FFNNnumCorpusSamples=${18}
FFNNOpt=${19}

stanOutputDir=${baseDir}"data/stanford_output_all/"

echo "-------- params --------"
echo "corpus:" ${corpusPath}
echo "useECBTest:" ${useECBTest} # 2
echo "onlyValidSentences:" ${onlyValidSentences} # 3
echo "numLayers:" $numLayers # 4
echo "numEpochs:" $numEpochs # 5
echo "windowSize:" $windowSize # 6
echo "numNegPerPos:" $numNegPerPos # 7
echo "batchSize:" $batchSize # 8
echo "dropout:" $dropout # 9
echo "numFilters:" $numFilters # 10
echo "filterMultiplier:" $filterMultiplier # 11
echo "posType:" $posType # 12
echo "posEmbeddingsFile:" $posEmbeddingsFile # static
echo "lemmaType:" $lemmaType # 13
echo "dependencyType:" $dependencyType # 14
echo "charType:" $charType # 15
echo "charEmbeddingsFile:" $charEmbeddingsFile # static
echo "devDir:" $devDir # 16
echo "FFNNnumEpochs:" $FFNNnumEpochs # 17
echo "FFNNnumCorpusSamples:" $FFNNnumCorpusSamples # 18
echo "FFNNOpt:" $FFNNOpt # 19

echo "-------- STATIC PATHS --------"
echo "resultsDir:" ${resultsDir}
echo "dataDir:" ${dataDir}
echo "verbose:" $verbose
echo "replacementsFile:" ${replacementsFile}
echo "embeddingsFile:" $embeddingsFile
echo "hddcrpFullFile:" $hddcrpFullFile
echo "stanOutputDir:" $stanOutputDir
echo "stanTokensFile:" $stanTokensFile
echo "verifiedSentencesFile:" $verifiedSentencesFile
echo "------------------------"

cd $scriptDir

python3 -u CorefEngine.py \
--corpusPath=${corpusPath} \
--useECBTest=${useECBTest} \
--onlyValidSentences=${onlyValidSentences} \
--stanTokensFile=${stanTokensFile} \
--verifiedSentencesFile=${verifiedSentencesFile} \
--replacementsFile=${replacementsFile} \
--verbose=${verbose} \
--stanOutputDir=${stanOutputDir} \
--baseDir=${baseDir} \
--hddcrpFullFile=${hddcrpFullFile} \
--numLayers=${numLayers} \
--embeddingsFile=${embeddingsFile} \
--numNegPerPos=${numNegPerPos} \
--numEpochs=${numEpochs} \
--batchSize=${batchSize} \
--windowSize=${windowSize} \
--dropout=${dropout} \
--numFilters=${numFilters} \
--filterMultiplier=${filterMultiplier} \
--posType=${posType} \
--posEmbeddingsFile=${posEmbeddingsFile} \
--lemmaType=${lemmaType} \
--dependencyType=${dependencyType} \
--charEmbeddingsFile=${charEmbeddingsFile} \
--charType=${charType} \
--devDir=${devDir} \
--FFNNnumEpochs=${FFNNnumEpochs} \
--FFNNnumCorpusSamples=${FFNNnumCorpusSamples} \
--FFNNOpt=${FFNNOpt}