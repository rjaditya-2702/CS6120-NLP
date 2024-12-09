from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import desc, row_number
from pyspark.sql.window import Window

# Set environment variables
JAVA_HOME = '/opt/homebrew/opt/openjdk'
SPARK_HOME = '/opt/homebrew/opt/apache-spark/libexec'

# Constants
CLASSES = {
    'LABEL_0': 'sadness',
    'LABEL_1': 'joy',
    'LABEL_2': 'love',
    'LABEL_3': 'anger',
    'LABEL_4': 'fear',
    'LABEL_5': 'surprise',
}

# Get the top 100 most confident samples from each class
def get_samples():
    # create a SparkSession
    spark = SparkSession.builder.appName("CSV to DataFrame").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()

    # define schema
    schema = StructType([
        StructField("index", StringType(), True),
        StructField("text", StringType(), True),
        StructField("label", StringType(), True),
        StructField("score", DoubleType(), True),
    ])

    # convert from csv to pyspark dataframe
    df = spark.read.format("csv").option("header", "true").schema(schema).load("code/test_inference.csv")

    # Partition by class and sort by score (descending)
    window = Window.partitionBy("label").orderBy(desc("score"))

    # Rank scores
    df = df.withColumn("rank", row_number().over(window))

    # Get the top 100 rows per class (100 highest scores)
    df = df.filter(df.rank <= 100)

    # Drop the rank column
    df = df.drop("rank")

    # Show the result
    df.show()

    # Save the samples from each class to new .csv files
    for label, class_name in CLASSES.items():
        sample = df.filter(df["label"] == str(label))
        sample.write.csv(f"code/samples/{class_name}", header=True, mode="overwrite")
