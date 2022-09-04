# n2c2 2022 shared task (Track 1) on Contextalized Medication Event Extraction (CMED)
The scripts are for the <a href="https://n2c2.dbmi.hms.harvard.edu/2022-track-1" target="_blank">n2c2 2022 CMED shared task</a>, Track 1, subtask 2 on medication event extraction.

The dataset was annotated using BRAT, and the task organizers shared the BRAT annotated files ('.ann' and '.txt' files).

The pre-processing script takes the folder of the BRAT files as input and outputs sequences of the form <i>[context] MEDICATION [context]</i>. 

The <i>[context]</i> can be of variable size, which is set using the <i>window</i> parameter.


