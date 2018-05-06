The program is made with Python 2.7.

The program can be run under three modes. UsualSetting flag is set
to one in the code when running it without analysis. ExploreModelChoice
is set to one when doing analysis 1 of the homework handout. And 
ExploreFeatureSpace flag is set to one when performing analysis 2.
At one time only one of the three flags must be set to one.

Here is how to run it with LR under UsualSettting = 1 flag.

$python hw3.py train.csv test.csv 1

and here is how to run it with SVM under UsualSetting = 1 flag.

$python hw3.py train.csv test.csv 2

If you want to run analysis 1 or 2 set the appropriate flag and do:

$python hw3.py data.csv