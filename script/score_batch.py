from pyspark.sql import SparkSession
import mlflow
from mlflow.pyfunc import load_model

spark = SparkSession.builder.appName("BatchScoring").getOrCreate()

test_df = spark.read.format("delta").load("/mnt/delta/test_data")


model = load_model("models:/AutoML_Best_Model/Production")


predictions = model.predict(test_df.toPandas())


pred_df = spark.createDataFrame(predictions)
pred_df.write.mode("overwrite").format("delta").save("/mnt/delta/predictions")
