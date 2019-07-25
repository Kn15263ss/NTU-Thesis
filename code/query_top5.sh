for file in `find top5_parameter -name "seed(50).csv"`; do
    t=$(dirname $(dirname $(dirname $file)))
    topk=$(basename $(dirname $file))
    model=$(basename $t)
    dataset=$(basename $(dirname $t))
    search_space_dirname="./Hyperband_32_results/${dataset}.${model}_result"
    search_space=$(find  "$search_space_dirname" -name "seed(50).csv")
    echo find parameter of $file
    pattern=`tail -n 1 $file | awk -F "," '{ NR==1; OFS="," } {print $3,$4,$5,$6}' -`

    result=$(echo $search_space | xargs -n 1 grep -rl $pattern)
    src_dir=$(dirname $result)
    mkdir -p $t/Hyperband_32/$topk
    cp -r $src_dir/* $t/Hyperband_32/$topk
done
