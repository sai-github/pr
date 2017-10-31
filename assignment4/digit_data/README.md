#### Dependent Packages
* sklearn
* numpy

#### Flow
* Run the `kmeans*.ipynb` notebook by doing KMeans on train data. 
* `d1.out`, `d2.out`, `d3.out` files are generated for a model in `hmm-1.04` directory
* Run `run.sh` for building and training each `d1` on `d1,d2,d3 hmm` models and similarly for `d2, d3`
* Accuracy of test class is in `hmm-1.04\hmm_alphas\acc.log`
