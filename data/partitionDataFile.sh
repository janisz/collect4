#!/usr/bin/env bash
full_filename="$1"
filename=$(basename $full_filename .csv)
filename_shuffled=$filename"_shuf.csv"
shuf $full_filename > $filename_shuffled
lines_count=$(wc -l < $full_filename)
one_fifth_lines=$(($lines_count/5))

sed -n '1,'$(($one_fifth_lines*3))'p' $filename_shuffled > $filename"_training.csv"
sed -n $(($one_fifth_lines*3))','$(($one_fifth_lines*4))'p' $filename_shuffled > $filename"_validation.csv"
sed -n $(($one_fifth_lines*4))','$(($lines_count))'p' $filename_shuffled > $filename"_test.csv"

rm $filename_shuffled

