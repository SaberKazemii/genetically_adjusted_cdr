# genetically_adjusted_cdr
Glaucoma is a leading reason for blindness and needs to be monitored and prevented. cup to disc ratio (CDR) is the important indicator of the glaucoma. The normal CDR is less than 0.6. However sometimes, this cutoff does not hold for glaucoma. Some patients with normal eye have higher than 0.6 CDR and vice versa. Hence we need to normalize the CDR.

Our apporoach in this project is to take the effect of genetic away from the CDR calculation. To this aim we train a deep learning model that fine tunes the ploygenic risk score (PRS) based on the Euoprpian ansectory groups to taeget anscestory groups. Then with the international classification of disease (ICD) codes, PRS and the raw CDR, we normalize the CDR. 
