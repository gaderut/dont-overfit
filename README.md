Aim: Tackle two of the most common practical problems observed within machine learning, the problem of small size of dataset and the problem of datasets skewed towards a single class both of which may lead to poor generalization and overfitting.

Packages used in this project:
pandas
numpy
scikit-learn
seaborn
matplotlib
collections

Python Version: python 3.7

To Run the training on Don't overfit II kaggle dataset run the following command:
python dont_overfit.py

Access to the MIMIC-III dataset can be obtained from : https://physionet.org/content/mimiciii/1.4/
The Dataset can be extracted from Google Cloud using Bigquery.
The new dataframes created are merged to form a single dataframe which is saved as csv file. The mimiciii.py file is used to run the code by setting the location of the csv file. 

To run the mimiciii.py file, mention the location of the csv file in the code.
and then execute the commmand python mimiciii.py
