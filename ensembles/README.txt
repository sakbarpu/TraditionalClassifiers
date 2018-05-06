The system can be run under different settings.
Please set the appropriate flag in the source code.

UsualSetting for usually running without analysis
ExploreDataSpace for analysis 1 
ExploreFeatureSpace 2
ExploreDepthSpace 3
ExploreTreeSpace 4

python hw4.py train.csv test.csv 1|2|3|4|5

1 for DT
2 for BT
3 for RF
4 for BS
5 for SVM

In the figures given in Analysis*.png files, you can see how the models behave as different parameters are changed:

1 Training set size
2 Feature set size
3 Depth limit (number of levels in the trees)
4 Number of trees
