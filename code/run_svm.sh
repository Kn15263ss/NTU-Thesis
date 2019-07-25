#!/usr/bin/env bash


while IFS= read -r file
do
	python3 svm.py $file
done < "./test_svm.txt"
