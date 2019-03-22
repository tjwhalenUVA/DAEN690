from pyspark import SparkContext 
sc = SparkContext.getOrCreate()

from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)

_wc = sqlCtx.read.json('C:/Users/e481340/Documents/GMU MASTERS/DAEN 690/DAEN690/pyspark/train_content.json')