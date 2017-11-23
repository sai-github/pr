#### Given mfcc features of 1,2,3 , model using SVM and ANNs

* Load all the necessary packages
* Load the data
* Find the reference utterance from each class 
* Reference utterance is that one whose length is maximum and common to all the class
* For all the utterances..do DTW with the reference utterance of the respective classes and find the path
* Using the indices in the path...shrink or expand your utterance and store it in 'X'
* Once you get the fixed length data..pass it to your SVM library
* Get the required scores and accuracy using different attributes of SVM library