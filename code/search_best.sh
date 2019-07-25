#/usr/bin/bash

# ###--- search accuracy($8) or loss($7) ---###
# for f in $(find . -name "seed(50).csv"); 
# 	do 
# 		echo "$f"
# 		cat "$f" | tail -n 1 | awk -F "," '{print $7}' ; 
# done

##--- search best parameter ---###
count()
{
	for f in $(find . -name "seed(50).csv"); 
	do 
			cat "$f" | tail -n 1 | awk -F "," '{for(i=3; i<7; i++) print $i}';
	done
}


for m in `ls .`
do
	if [ -d "$m" ]; then
		cd $m
		for dm in `ls .`
		do
			if [ -d "$dm" ]; then
				cd $dm
				for ddm in `ls .`
				do
					if [ -d "$ddm" ] && [ "$ddm" = "randomsearch_8" ] || [ "$ddm" = "randomsearch_16" ]; then
					# if [ -d "$ddm" ]; then
						cd $ddm
						echo $m $dm $ddm
						count
						cd ..
					fi
				done
				cd ..
			fi
		done
		cd ..
	fi
done