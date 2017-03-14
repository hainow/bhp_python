# build confusion matrix 
# argv1: label file 
# argv2: directory holding a list of files, each file stores the result for OCR test, named after the label name

import sys
import os
import csv 
from collections import OrderedDict


#matrix = {}
matrix = OrderedDict() 
labels = []

# build the matrix first 
with open(sys.argv[1], 'r') as label_file: 
	for line in label_file: 
		labels.append(line.strip('\n'))

num_labels = len(labels)

# create a scaffold for matrix
for word in labels:
	matrix[word] = OrderedDict() 

for i in labels:
	for j in labels: 
		matrix[i].update({j:0})

# print matrix # debug purpose only 

# read all files in the directory provided by sys.argv2
for dirpath,_,filenames in os.walk(sys.argv[2]):
   for fname in filenames:
		# retrieve the correct word (label) by filenames
		correctWord = fname.split('.')[0]
		
		fullPath = os.path.abspath(os.path.join(dirpath, fname))
		with open(fullPath, 'r') as f: 
			for line in f: 
				fields = line.strip('\n').split('\t')
				if fields[0] == 'Script': 
					continue
                         	
				# update matrix
				matrix[correctWord][fields[1]] += 1 


# write to output as CSV file 
result_file = csv.writer(open("confusion_matrix.csv", "w"))
result_file.writerows([[' '] + labels])
result_file2 = csv.writer(open("confusion_matrix2.csv", "w"))
result_file2.writerows([[' '] + labels])

for k, v in matrix.items():
	predicted_words = [k]
        predicted_words2 = []
	#predicted_words = []
	for k2, v2 in v.items(): 
		predicted_words.append({k2:v2})
                predicted_words2.append(v2)
	result_file.writerows([predicted_words])
        result_file2.writerows([predicted_words2])

