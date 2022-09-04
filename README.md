# n2c2 2022 shared task on Contextalized Medication Event Extraction (CMED)
The scripts are for the <a href="https://n2c2.dbmi.hms.harvard.edu/2022-track-1" target="_blank">n2c2 2022 CMED shared task</a> Track 1, subtask 2 on medication event extraction.

The dataset was annotated using BRAT, and the task organizers provided the BRAT annotated files ('.ann' and '.txt' files).

The pre-processing script takes the folder of the BRAT files as input and outputs sequences of the form <i>[context] MEDICATION [context]</i>. 
The size of the <i>[context]</i> is variable, and can be set using the <i>window</i> parameter.

For more details on this task check my <a href="" target="_blank">projects website</a> and the <a href="https://n2c2.dbmi.hms.harvard.edu/2022-track-1" target="_blank">n2c2 website</a>.

