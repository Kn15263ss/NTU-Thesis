#!/usr/bin/env bash

while IFS= read -r file
do
       python3 main.py $file
done < "./test.txt"
