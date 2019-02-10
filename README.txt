[1] To run Stanford Dependency Parser on all of our files:
  (a) on .xml files which have 1 line per sentence: ls -1 | xargs readlink -f > all-files.txt
  (b) run stanford: java -cp "*" -Xmx10g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -filelist ../../CRETE/data/stanford_in/all-files.txt -outputDirectory out -ssplit.eolonly -tokenize.options untokenizable=noneKeep -parse.debug
