#!/bin/bash
prefix="ccnnagg" # used to help identify experiments' outputs, as the output files will have this prefix
corpus="FULL"
useECBTest=true
featureMap=(2) # 4)
numLayers=(2) # 3) # 1 3
numEpochs=(3) # 20)
windowSize=(0)
numNeg=(5)
batchSize=(128) # 128) # 64 128
dropout=(0.0) # 0.2 0.4)
numFilters=(64)
filterMultiplier=(1.0) # 2.0)
posType=("none") # none  sum  avg
lemmaType=("none") # "sum" "avg")
dependencyType=("none") # # "sum" "avg")
charType=("none") # "none" "concat" "sum" "avg"
devDir=(23) # this # and above will be the dev dirs.  See ECBHelper.py for more

hn=`hostname`

# FEATURE MAP OVERRIDE
if [[ " ${featureMap[*]} " == *"1"* ]]; then
	posType=("sum")
	prefix=${prefix}1
fi
if [[ " ${featureMap[*]} " == *"2"* ]]; then
	lemmaType=("sum")
	prefix=${prefix}2
fi
if [[ " ${featureMap[*]} " == *"3"* ]]; then
	dependencyType=("sum")
	prefix=${prefix}3
fi
if [[ " ${featureMap[*]} " == *"4"* ]]; then
	charType=("concat")
	prefix=${prefix}4
fi

# FFNN params
FFNNnumEpochs=(10)
FFNNnumCorpusSamples=(1) # 5 10 20)
FFNNPosRatio=(0.8) # 0.2 0.8
FFNNOpt=("adam") # "rms" "adam" "adagrad"
source ~/researchcode/venv/bin/activate

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
								for pt in "${posType[@]}"
								do
									for lt in "${lemmaType[@]}"
									do
										for dt in "${dependencyType[@]}"
										do
											for ct in "${charType[@]}"
											do
												for dd in "${devDir[@]}"
												do
													for fn in "${FFNNnumEpochs[@]}"
													do
														for fp in "${FFNNnumCorpusSamples[@]}"
														do
															for fo in "${FFNNOpt[@]}"
															do
																# qsub -pe smp 8 -l vlong -o

																fout=gpu_${prefix}_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_dr${dr}_nf${nf}_fm${fm}_pt${pt}_lt${lt}_dt${dt}_ct${ct}_dd${dd}_fn${fn}_fp${fp}_fo${fo}.out
																echo ${fout}
																if [ ${hn} = "titanx" ] || [ ${hn} = "Christophers-MacBook-Pro-2" ]
																then
																	echo "* kicking off CRETE2 natively"
																	./CRETE2.sh ${corpus} ${useECBTest} ${nl} ${ne} ${ws} ${neg} ${bs} ${dr} ${nf} ${fm} ${pt} ${lt} ${dt} ${ct} ${dd} ${fn} ${fp} ${fo} # > ${fout}												
																else
																	qsub -l gpus=1 -o ${fout} CRETE2.sh ${corpus} ${useECBTest} ${nl} ${ne} ${ws} ${neg} ${bs} ${dr} ${nf} ${fm} ${pt} ${lt} ${dt} ${ct} ${dd} ${fn} ${fp} ${fo}
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
					done
				done
			done
		done
	done
done