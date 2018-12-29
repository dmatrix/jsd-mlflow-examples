import mlflow.spark
import pyspark

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

if __name__ == '__main__':

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    training = spark.createDataFrame([ (0, "a b c d e spark", 1.0),
                    (1, "b d", 0.0),
                    (2, "spark f g h", 1.0),
                    (3, "hadoop mapreduce", 0.0),
                    (4, "mlflow spark integration", 1.0),
                    (5, "try spark udf", 1.0)], [str(1), str(2), "label"])
    tokenizer = Tokenizer(inputCol=str(2), outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    model = pipeline.fit(training)

    mlflow.set_experiment("v0.8.1-udf")
    with mlflow.start_run():
        mlflow.spark.log_model(model, "spark-model")
        mlflow.log_param("keyword", "spark")
        mlflow.log_param("keys", training.count())

    print("Done Running Experiment v0.8.1-udf")