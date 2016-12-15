# [Bosch production line performance](https://www.kaggle.com/c/bosch-production-line-performance)


#Exploratory data analsis based on PySpark RDD based API and Jupyter notebook.

Decision_Tree_Refine.ipynb	

#Running the code 
Requires $SPARK_HOME to be set . Can be run directly on desktop with a PySpark standalone mode. Worked fine for me with out even setting up master and workers! 
$ nohup python script.py &

#Feature reduction
Feature reduction (/information) on numeric data can be calculated with "feature_correlation.py". This would output a list of highly correlated  columns to be removed (using Pearson correlation criterion ) and finally save a reduced feature dataframe that could be used in SPARK ML or MLlib.

#Decision tree / Random Forest / GBT
Once we have the information for the list of columns to be removed then we can invoke MLlib (RDD based machine learning in Spark)) or  ML (dataframe based machine learning in Spark). The data set is split in to (0.7,0.3) ratio for traning and test set respecteively . The test set predictions are accessed based on :Accuracy , confusion matrix, and Matthews correlation coefficient . 

Decision_Tree_Reduced_Feature.py
Random_forest_reduced_feature.py 
