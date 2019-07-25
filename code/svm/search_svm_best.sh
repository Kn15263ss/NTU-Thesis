#/usr/bin/bash


for f in $(find . -name "seed(50).csv"); 
	do 
		echo "$f"
		cat "$f" | tail -n 1 | awk -F "," '{print $NF}' ; 
done

