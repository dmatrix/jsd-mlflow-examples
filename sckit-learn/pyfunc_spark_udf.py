import pyspark

from pyspark.sql.types import StringType

from mlflow.pyfunc import spark_udf

if __name__ == '__main__':

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame([
                (4, "spark i j k"),
                (5, "l m n"),
                (6, "spark hadoop spark"),
                (7, "apache hadoop")], [str(1), str(2)])
    pyfunc_udf = spark_udf(spark, "spark-model", "f2ccde5b33ce456d973ce9f91de8cadf", result_type=StringType())
    new_df = spark_df.withColumn("prediction", pyfunc_udf(str(1), str(2)))
    new_df.show()