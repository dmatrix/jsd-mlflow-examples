import pyspark
import os

from pyspark.sql.types import DoubleType
from sklearn.model_selection import train_test_split

from mlflow.pyfunc import spark_udf
import pandas as pd

if __name__ == '__main__':

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    test_y = test[["quality"]]

    pdf = pd.DataFrame(test_y)
    spark_df = spark.createDataFrame(pdf)
    pyfunc_udf = spark_udf(spark, "model", "3774808880c14057abcc89106caa70f9", result_type=DoubleType())
    new_df = spark_df.withColumn("prediction", pyfunc_udf("quality"))
    new_df.show()