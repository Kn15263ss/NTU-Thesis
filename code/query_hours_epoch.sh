#/usr/bin/bash

### ---- calculate hours ---- ###
count() {
	s=0
	for f in $(find . -name "result.json");
	do
		t=$(cat $f | tail -n 1 | grep -o -P '.{1}time_total_s.{0,20}' | grep -Eo '[0-9]+([.][0-9]+)?')
		s=`echo $s + $t | bc`
	done
	echo "scale=1; $s / 3600.0" | bc
}
count


for m in `ls .`
do
	if [ -d "$m" ] && [ "$m" != 'test_nn' ] && [ "$m" != 'best_parameter' ]; then
		cd $m
		for dm in `ls .`
		do
			if [ -d "$dm" ]; then
				cd $dm
				echo $m $dm
				count
				cd ..
			fi
		done
		cd ..
	fi
done

### ---- calculate epoch ---- ###
# count() 
# {
# 	s=0
# 	for f in $(find . -name "result.json");
# 	do
# 		if [ -s $f ]
# 		then 
# 			t=$(cat $f | tail -n 1 | grep -o -P '.{1}epoch.{0,5}' | grep -Eo '[0-9]+()?')
# 		else
# 			t=1
# 		fi
# 		s=`echo $s + $t | bc`
# 	done
# 	echo $s
# }

# for m in `ls .`
# do
# 	if [ -d "$m" ] && [ "$m" != 'test_nn' ] && [ "$m" != 'best_parameter' ]; then
# 		cd $m
# 		for dm in `ls .`
# 		do
# 			if [ -d "$dm" ]; then
# 				cd $dm
# 				echo $m $dm
# 				count
# 				cd ..
# 			fi
# 		done
# 		cd ..
# 	fi
# done