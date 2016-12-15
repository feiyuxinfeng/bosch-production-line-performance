# [Bosch production line performance](https://www.kaggle.com/c/bosch-production-line-performance)


##Exploratory data analsis based on PySpark RDD based API and Jupyter notebook.

* [Decision_Tree_Refine.ipynb](https://github.com/pythonpanda/bosch-production-line-performance/blob/master/Decision_Tree_Refine.ipynb)	

##Running the code 
Requires $SPARK_HOME to be set . Can be run directly on desktop with a PySpark standalone mode. Worked fine for me with out even setting up master and workers! 
$ nohup python script.py &

##Feature reduction
Feature reduction (/information) on numeric data has been performed by exploiting the high level Spark dataframe ([SparkSQL](https://spark.apache.org/docs/1.6.2/api/python/pyspark.sql.html)):

* [feature_correlation.py](https://github.com/pythonpanda/bosch-production-line-performance/blob/master/feature_correlation.py) 

Highly correlated feature information is constructed based on [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) criterion. The reduced features can be used for training SPARK ML or MLlib.

##Decision tree / Random Forest / GBT
Once we have the information for the list of columns to be removed then, we can invoke MLlib (RDD based machine learning in Spark)) or  ML (dataframe based machine learning in Spark). The data set is split in to (0.7,0.3) ratio for traning and test set respecteively . The test set predictions are accessed based on : 
* Accuracy , [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), and [Matthews correlation coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) . 

* [Decision_Tree_Reduced_Feature.py](https://github.com/pythonpanda/bosch-production-line-performance/blob/master/Decision_Tree_Reduced_Feature.py)
* [Random_forest_reduced_feature.py](https://github.com/pythonpanda/bosch-production-line-performance/blob/master/Random_forest_reduced_feature.py) 
