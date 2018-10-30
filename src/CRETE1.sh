#!/bin/bash
export PYTHONIOENCODING=UTF-8
prefix="ccnn" # used to help identify experiments' outputs, as the output files will have this prefix
corpus="FULL"
onlyValidSentences="T"
addIntraDocs="F" # since these are singletons w.r.t. cross-doc
exhaustivelyTestAllFeatures=false
useECBTest=true
featureMap=(2 3) # 1 2 3 4 5 6 7)
numLayers=(2) # 3) # 1 3
numEpochs=(10) # 20)
windowSize=(0)
numNeg=(5)
batchSize=(64) # 128) # 64 128
dropout=(0.0) # 0.2 0.4)
numFilters=(32)
filterMultiplier=(1.0) # 2.0)
devDir=(23) # this # and above will be the dev dirs.  See ECBHelper.py for more

# features (default = False)
wordFeature="False" # f1
lemmaFeature="False" # f2
charFeature="False" # f3
posFeature="False" # f4
dependencyFeature="False" # f5
bowFeature="False" # f6
wordnetFeature="False" # f7
framenetFeature="False" # f8

native="False"
hn=`hostname`

IFS=$'\r\n' GLOBIGNORE='*' command eval  'XYZ=($(cat featureCombos.txt))'
for features in "${XYZ[@]}"
do
	prefix="ffnn"
	if [ "$exhaustivelyTestAllFeatures" = false ] ; then
		features=${featureMap[@]}
	fi
	# FEATURE MAP OVERRIDE
	if [[ " ${features[*]} " == *"1"* ]]; then
		wordFeature="T"
		prefix=${prefix}1
	fi
	if [[ " ${features[*]} " == *"2"* ]]; then
		lemmaFeature="T"
		prefix=${prefix}2
	fi
	if [[ " ${features[*]} " == *"3"* ]]; then
		charFeature="T"
		prefix=${prefix}3
	fi
	if [[ " ${features[*]} " == *"4"* ]]; then
		posFeature="T"
		prefix=${prefix}4
	fi
	if [[ " ${features[*]} " == *"5"* ]]; then
		dependencyFeature="T"
		prefix=${prefix}5
	fi
	if [[ " ${features[*]} " == *"6"* ]]; then
		bowFeature="T"
		prefix=${prefix}6
	fi
	if [[ " ${features[*]} " == *"7"* ]]; then
		wordnetFeature="T"
		prefix=${prefix}7
	fi
	if [[ " ${features[*]} " == *"8"* ]]; then
		framenetFeature="T"
		prefix=${prefix}8
	fi
	# FFNN params
	FFNNnumEpochs=(10)
	FFNNPosRatio=(0.8) # 0.2 0.8
	#source ~/researchcode/venv/bin/activate
	for nl in "${numLayers[@]}"
	do
		for ne in "${numEpochs[@]}"
		do
			for ws in "${windowSize[@]}"
			do
				for neg in "${numNeg[@]}"
				do
					for bs in "${batchSize[@]}"
					do
						for dr in "${dropout[@]}"
						do
							for nf in "${numFilters[@]}"
							do
								for fm in "${filterMultiplier[@]}"
								do
									for dd in "${devDir[@]}"
									do
										for fn in "${FFNNnumEpochs[@]}"
										do
											# qsub -pe smp 8 -l vlong -o
											fout=gpu_${prefix}_ov${onlyValidSentences}_id${addIntraDocs}_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_dr${dr}_nf${nf}_fm${fm}_dd${dd}_fn${fn}.out
											echo ${fout}
											if [ ${hn} = "ctanner" ] || [ ${hn} = "Christophers-MacBook-Pro-2" ]
											then
												echo "* kicking off CRETE2 natively"
												native="True"
												./CRETE2.sh ${corpus} ${useECBTest} ${onlyValidSentences} ${addIntraDocs} ${nl} ${ne} ${ws} ${neg} ${bs} ${dr} ${nf} ${fm} ${wordFeature} ${lemmaFeature} ${charFeature} ${posFeature} ${dependencyFeature} ${bowFeature} ${wordnetFeature} ${framenetFeature} ${dd} ${fn} ${native}
											else
												qsub -l gpus=1 -o ${fout} CRETE2.sh ${corpus} ${useECBTest} ${onlyValidSentences} ${addIntraDocs} ${nl} ${ne} ${ws} ${neg} ${bs} ${dr} ${nf} ${fm} ${wordFeature} ${lemmaFeature} ${charFeature} ${posFeature} ${dependencyFeature} ${bowFeature} ${wordnetFeature} ${framenetFeature} ${dd} ${fn} ${native}
											fi
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
	if [ "$exhaustivelyTestAllFeatures" = false ] ; then
		exit 1
	fi
done