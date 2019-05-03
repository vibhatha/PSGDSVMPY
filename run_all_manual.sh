#for alpha in `seq 1 2 50`;
declare -a arr=("0.000001" "0.00001" "0.0001" "0.001" "0.01" "0.1" "1")

## now loop through the above array
for i in "${arr[@]}"
do
   sh run_manual.sh $i
   # or do whatever with individual element of the array
done
echo "All Manual Job done" | mail -s "SGD SVN: Job notice" vibhatha@gmail.com

