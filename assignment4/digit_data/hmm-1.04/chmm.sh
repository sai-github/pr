files=$( ls chmms/chmms_m/*.hmm )
counter=0
for i in $files
do
  let counter=$counter+1
  ./test_hmm "d1_connected_test.out" $i
  temp="$(cut -d'/' -f3 <<<"$i")"
  cp alphaout "alphas_connected/"$temp".alpha"
done

python alphas_connected/pred_connected.py
