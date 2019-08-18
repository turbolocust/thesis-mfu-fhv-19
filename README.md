# thesis-mfu-fhv-19

Data is provided only for the anonymized dataset.
Paths in scripts and batch files may need to be adjusted.

The structure of this repository is as follows:

 - **code**
	 - **pyml**
		 - Contains Python scripts and batch files.
	 - **abap**
		 - Contains ABAP reports and classes.
 - **nmt**
	 - Contains a fork of the NMT Tutorial by TensorFlow.
	 - See: https://github.com/tensorflow/nmt
 - **data**
	 - **charts**
		 - Contains diagrams created by *code/plot_log.py*
	 - **comp**
		 - Contains comparison files created by *code/file_compare.py* and *code/stat_collector.py* for all models.
	 - **models**
		 - Contains trained models or mirrors to trained models.
	 - **ngrammer**
		 - Contains raw output of *code/abap/zreswo_ngram.asprog*
	 - **post_analysis**
		 - Contains files for the post analysis and files created by *code/post_analyze.py*
	 - **vocab**
		 - Contains the whitelisted source and target vocabulary.
		 - **glove**
			 - Contains the target vocabulary in glove format.
	 - **workbooks**
		 - Contains spreadsheet files with more detailed statistics.

---
**b4** indicates batch number four. The following applies:
 - k5 folds were randomly shuffled by lines.
 - k10 folds were randomly shuffled by documents.
<br />

**b5** indicates batch number five. The following applies:
 - k5 folds were randomly shuffled by documents.
