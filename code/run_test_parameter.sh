#!/usr/bin/env bash


while IFS= read -r file
do
	python3 test_parameter.py $file
done < "./test_2.txt"
