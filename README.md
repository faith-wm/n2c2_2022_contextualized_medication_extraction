# n2c2 2022 shared task on Contextalized Medication Event Extraction (CMED)
The scripts are for the <a href="https://n2c2.dbmi.hms.harvard.edu/2022-track-1" target="_blank">n2c2 2022 CMED shared task</a> Track 1, subtask 2 on medication event extraction.

The dataset was annotated using BRAT, and the task organizers provided the BRAT annotated files ('.ann' and '.txt' files).

The pre-processing script takes the folder of the BRAT files as input and outputs sequences of the form <i>[context] MEDICATION [context]</i>. 
The size of the <i>[context]</i> is variable, and can be set using the <i>sorrounding_chars</i> parameter.

For more details on this task check my <a href="https://faith-wm.github.io/cmed_proj.html" target="_blank">projects website</a> and the <a href="https://n2c2.dbmi.hms.harvard.edu/2022-track-1" target="_blank">n2c2 website</a>.

# Models 
BioClinical BERT: <a href="Bio_ClinicalBERT">https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT</a>

Clinical Longformer: <a href="https://huggingface.co/yikuan8/Clinical-Longformer">https://huggingface.co/yikuan8/Clinical-Longformer</a>

PubMed BERT: <a href="https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract">https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract</a>
