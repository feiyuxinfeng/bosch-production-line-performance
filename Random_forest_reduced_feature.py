#!$SPARK_HOME/bin/pyspark
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from time import time
from numpy import NaN
from numpy import array
from numpy import sqrt
import  csv
try:
    sc.stop()
except:
    pass
sc=SparkContext('local','pyspark')
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

"""
A pyspark Script to perform feature reduction for Bosch production line numeric RDD.
Performs Decision Tree classification on reduced data
"""


def labelData(data):
    """ A function to Construct LabeledPoint for Spark MLlib supervised learning """
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[1:-1]))

def getLabelsPredictions(model, test_data):
    """ A function to compute predictions from model and test data"""
    predictions = model.predict(test_data.map(lambda p: p.features))
    labels_and_preds = test_data.map(lambda p: p.label).zip(predictions)
    return labels_and_preds

def printMCC(labels_and_preds):
    """ A function to get  confusion matrix and MCC """

    TP = labels_and_preds.filter(lambda (v, p): int(v) == 1 and  int(p) == 1 ).count()
    FP = labels_and_preds.filter(lambda (v, p): int(v) == 0 and  int(p)  == 1 ).count()
    FN = labels_and_preds.filter(lambda (v, p): int(v) == 1 and  int(p) == 0 ).count()
    TN = labels_and_preds.filter(lambda (v, p): int(v) == 0 and  int(p) == 0 ).count()
    MCC = float(TP*TN - FP*FN)/float(sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)   ) )
    print('MCC Score   : %.4f      '%(MCC))
    CM = array([[TP,FN],[FP,TN]])
    print('Confusion Matrix\n')
    print(CM)


# Gather our code in a main() function
def main():
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
    df.na.fill(NaN)
    #Prepare feature for computation!
    #Remove features from a list precompiled on correlation criterion!
    counter = 0
    with open('column_refine_list.csv','r') as f:
        csvlist = csv.reader(f,delimiter=',')
        for item in csvlist:
            column_to_go = item[:]
    print("Total numer of features to be removed: %d"%(len(column_to_go)))
    print("\n")
    for item in column_to_go:
        df = df.drop(item)
    print("Final number of features is: %d"%(len(df.columns[1:-1])))
    #Decision Tree model Training
    training_points = labelData(df) #csv_numeric.map(ReducedlabelData)
    training_data, test_data = training_points.randomSplit([0.7,0.3])
    print(training_data.first())
    t0= time()
    tree_model = RandomForest.trainClassifier(training_data,numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=6, featureSubsetStrategy="auto",
                                     impurity='gini',maxDepth=8,maxBins=100)
    tt= time() - t0
    tree_model.save(sc,"RFmodel_model_reduced_NaN")  #Save the model for future use!
    print("Model trained in : %.4f Sec"%(tt))
    print(tree_model.toDebugString())

    #Making predictions on the test set
    ## Predict
    t0 = time()
    labels_and_preds = getLabelsPredictions(tree_model,test_data)
    test_accuracy = 100*labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
    print("Test accuracy is    : %.4f "%(test_accuracy))
    printMCC(labels_and_preds)
    tt = time() - t0
    print("Predictions and metrics computed in : %.4f Sec"%(tt))
    sc.stop()

if __name__ == '__main__':
    main()
