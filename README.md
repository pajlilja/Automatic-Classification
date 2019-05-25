# AutomaticClassification
Code base for the study "Automatic classification of neurons by their morphology"

## Url to the thesis
http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1214317&dswid=-2523
 
# Code
## getData.py
Gathers the data from neuromorpho.org

## transformData.py
Transform the gathered data so that is suites our
classifiers.

## classifier_assessment.py
Creates the classifiers using sklearn, runs the tests and
displays the results.

## cell_type_distribution.py
Plots the cell type distribution.

## cell_type_accuracy_distribution.py
Plots the cell type accuracy distribution.

# Data
## after1000runs.json
This file contains the result from 1000 runs on each of the
classifiers.

For each classifier each type is stored together with
the list of the amount of correctly classified types and
of the amount of missclassified types. 
E.g. 
```code
{'Purkinje': [11160, 3441]}
```

# Images
The images used for the report.
