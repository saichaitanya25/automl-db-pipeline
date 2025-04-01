from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName("DeltaIngest").getOrCreate()

df = spark.read.csv("/mnt/raw/input.csv", header=True, inferSchema=True)


df.write.format("delta").mode("overwrite").save("/mnt/delta/training_data")
print("Data written to Delta Lake at /mnt/delta/training_data")
