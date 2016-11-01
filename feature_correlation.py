#!$SPARK_HOME/bin/pyspark
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkContext
from numpy import NaN
try:
    sc.stop()
except:
    pass
sc=SparkContext('local','pyspark')
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
import csv
from numpy import floor
"""
A pyspark Script to perform feature reduction for Bosch production line numeric data.
Performs feature selection based on pair-wise pearson correlation 
"""

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PythonSQL")\
        .config("spark.some.config.option", "some-value")\
        .getOrCreate()
    #Import a messed up  DataFrame (DF) to get Column info
    raw_df= sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferchema='true') \
    .load("/data/ganesh/BigData/Bosch/Source/train_numeric.csv.gz")
    features = raw_df.columns
    #Manually specify the correct datatypes for each Column and import new DF
    fields = [StructField(field_name, FloatType(), True) for field_name in features]
    fields[0].dataType = IntegerType()
    fields[-1].dataType = IntegerType()
    customSchema = StructType(fields)
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true') \
    .load("/data/ganesh/BigData/Bosch/Source/train_numeric.csv.gz",schema = customSchema)
    df_copy = sqlContext.read.format('com.databricks.spark.csv').options(header='true') \
    .load("/data/ganesh/BigData/Bosch/Source/train_numeric.csv.gz",schema = customSchema)
    df.fillna(NaN)
    df_copy.fillna(NaN)
    #Prepare feature for computation!
    features.remove('Id')
    features.remove('Response')
    feature_length = range(len(features))
    sampled_df = df.select(df.columns).sample(False, 0.10)
    #Avoid double counting features (including self counting) while computing pair-wise terms-- Two nearest neighbour chosen!
    counter = 0
    column_to_go = []
    for iter in feature_length[:-10]:
        if iter < feature_length[-11]:
            for num in range(iter,iter+12):
                if iter != num:
                    print("Computing correlation coefficient for :%d , %d "%(iter,num))
                    print("\n")
                    p_cor = sampled_df.stat.corr(features[iter],features[num])
                    print(features[iter],features[num])
                    print("The pair-wise correlation  is : %.4f"%(p_cor))
                    print("\n")
                    if abs(p_cor) > 0.8:
                        if features[iter] not in column_to_go:
                            column_to_go.append(features[iter])
                            counter += 1
                            print("%s has been removed from the DF"%(features[num]))
        else:
            for  iter in  feature_length[-11:-1]:
                for num in range(iter,len(features)):
                    if iter != num:
                        p_cor=sampled_df.stat.corr(features[iter],features[num])
                        print("The pair-wise correlation  is : %.4f"%(p_cor))
                        print("\n")
                        if abs(p_cor) > 0.8:
                            if features[iter] not in column_to_go:
                                column_to_go.append(features[iter])
                                counter += 1
                                print("%s has been removed from the DF"%(features[num]))

    print("Total numer of features removed: %d"%(counter))
    #Write new column header to a new list
    myfile = open('/data/ganesh/BigData/Bosch/Source/feature_reduction/numeric/column_refine_list.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(column_to_go)
    myfile.close()
    print("\n")
    print(column_to_go)
    # Write the new DF to a spark compressed CSV
    #df_copy.select(df.columns).write.format('com.databricks.spark.csv') \
    #.options(codec="org.apache.hadoop.io.compress.GzipCodec") \
    #.save('/data/ganesh/BigData/Bosch/Source/feature_reduction/numeric/reduced_numerc.csv')
    sc.stop()
