# bosch-production-line-performance
https://www.kaggle.com/c/bosch-production-line-performance
#Running the code 
Requires $SPARK_HOME to be set . Can be run directly on desktop with a PySpark standalone mode. Worked fine for me with out even setting up master and workers! 
$ nohup python script.py &
#Feature reduction
Feature reduction on numeric data can be done with "feature_correlation.py". This would output a list of columns to be removed and also a save a reduced feature dataframe that could be used in SPARK ML or MLlib.
#Decision tree / Random Forest / GBT
