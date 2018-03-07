#!/bin/bash
export PYTHONIOENCODING=UTF-8

# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/DeepCoref/"
brownDir="/home/ctanner/researchcode/DeepCoref/"

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
charEmbeddingsFile=${baseDir}"data/charRandomEmbeddings.txt"

stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"

corpusPath=${baseDir}"data/ECB_$1/"
useECBTest=$2
numLayers=$3
numEpochs=$4
windowSize=$5
numNegPerPos=$6
batchSize=$7

# CCNN features 
dropout=${8}
numFilters=${9}
filterMultiplier=${10}
posType=${11}
posEmbeddingsFile=${baseDir}"data/posEmbeddings100.txt"
lemmaType=${12}
dependencyType=${13}
charType=${14}
devDir=${15}
FFNNnumEpochs=${16}
FFNNnumCorpusSamples=${17}
FFNNOpt=${18}

stanOutputDir=${baseDir}"data/stanford_output/"

echo "-------- params --------"
echo "corpus:" ${corpusPath}
echo "useECBTest:" ${useECBTest} # 2 
echo "numLayers:" $numLayers # 3
echo "numEpochs:" $numEpochs # 4
echo "windowSize:" $windowSize # 5
echo "numNegPerPos:" $numNegPerPos # 6
echo "batchSize:" $batchSize # 7
echo "dropout:" $dropout # 8
echo "numFilters:" $numFilters # 9
echo "filterMultiplier:" $filterMultiplier # 10
echo "posType:" $posType # 11
echo "posEmbeddingsFile:" $posEmbeddingsFile # static
echo "lemmaType:" $lemmaType # 12
echo "dependencyType:" $dependencyType # 13
echo "charType:" $charType # 14
echo "charEmbeddingsFile:" $charEmbeddingsFile # static
echo "devDir:" $devDir # 15
echo "FFNNnumEpochs:" $FFNNnumEpochs # 16
echo "FFNNnumCorpusSamples:" $FFNNnumCorpusSamples # 17
echo "FFNNOpt:" $FFNNOpt # 18

echo "-------- STATIC PATHS --------"
echo "resultsDir:" ${resultsDir}
echo "dataDir:" ${dataDir}
echo "verbose:" $verbose
echo "replacementsFile:" ${replacementsFile}
echo "embeddingsFile:" $embeddingsFile
echo "hddcrpFullFile:" $hddcrpFullFile
echo "stanOutputDir:" $stanOutputDir
echo "------------------------"

cd $scriptDir