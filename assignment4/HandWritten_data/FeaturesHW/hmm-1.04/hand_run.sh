make clean
make all

symbol=16
state=6
./train_hmm d1.out 1234 $state $symbol 0.01
./train_hmm d2.out 1234 $state $symbol 0.01
./train_hmm d3.out 1234 $state $symbol 0.01

count=0
declare -a filename=("d1_test.out" "d2_test.out" "d3_test.out")
for i in "${filename[@]}"
do
  ./test_hmm $i d1.out.hmm
  cp alphaout "hmm_alphas/1.out"
  ./test_hmm $i d2.out.hmm
  cp alphaout "hmm_alphas/2.out"
  ./test_hmm $i d3.out.hmm
  cp alphaout "hmm_alphas/3.out"

  count=$((count+1))
  python "hmm_alphas/pred.py" $count
done
