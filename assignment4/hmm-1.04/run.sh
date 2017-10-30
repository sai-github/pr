make clean
make all

symbol = 16
state = 6
./train_hmm d1.out 1234 $symbol $state 0.01
./train_hmm d2.out 1234 $symbol $state 0.01
./train_hmm d3.out 1234 $symbol $state 0.01


./test_hmm d1.out d1.out.hmm
cp alphaout "hmm_alphas/1/1.out"
./test_hmm d1.out d2.out.hmm
cp alphaout "hmm_alphas/1/2.out"
./test_hmm d1.out d3.out.hmm
cp alphaout "hmm_alphas/1/3.out"

python "hmm_alphas/pred.py"
