		sns.set()
		d = {}
		labels = []
		xlabels = []
		countToREFREF = defaultdict(set)
		REFREFToPercent = {}
		for rel1 in sorted(relationshipTypes):
			col = []
			xlabels.append(rel1)
			cur_labels = []
			for rel2 in sorted(relationshipTypes):
				num_coref = withinDocRelRelCoref[rel1][rel2]["eventCoref_yes"]
				num_notcoref = withinDocRelRelCoref[rel1][rel2]["eventCoref_no"]
				total = num_coref + num_notcoref

				percent = 0
				if num_coref + num_notcoref > 0:
					percent = round(100*num_coref / (total), 0)
				if rel1 <= rel2:
					countToREFREF[total].add((rel1, rel2))
					REFREFToPercent[(rel1, rel2)] = percent
				cur_labels.append(str(percent) + " (" + str(total) + ")")
				col.append(percent) # data
			labels.append(cur_labels)
			d[rel1] = col

		for key in sorted(countToREFREF, reverse=True):
			for relpair in countToREFREF[key]:
				print(relpair,str(key),"counts;",str(REFREFToPercent[relpair]),"% coref")
		labels = np.array(labels)
		#print("l:", labels)
		#d = {'col1': [5, 8, 17], 'col2': [8, 6, 3], 'col3': [17, 4, 3]}
		df = pd.DataFrame(data=d)
		#xlabels = ['ab', 'b', 'c']
		#ylabels = ['x', 'y', 'z']
		#labels =  np.array([['A','B','C'],['C','D','E'],['E','F','G']])
		#print(xlabels)
		ax = plt.axes()
		sns.set_context("notebook", font_scale=0.75)
		sns.heatmap(df, ax=ax, annot=labels, xticklabels=xlabels, yticklabels=xlabels, cmap="Blues", fmt = '', )
		#ax.set_title('WD-Doc: % of Coref per Governor Dependency Relation Pairs')
		plt.show()
