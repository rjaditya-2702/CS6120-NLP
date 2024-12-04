from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import desc, row_number, split, col, explode, lit
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.window import Window
import os
JAVA_HOME = '/opt/homebrew/opt/openjdk'
SPARK_HOME = '/opt/homebrew/opt/apache-spark/libexec'
os.environ["PYSPARK_PYTHON"] = "/Users/neetidesai/anaconda3/envs/nlp-hw2-venv/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Users/neetidesai/anaconda3/envs/nlp-hw2-venv/bin/python"

CLASSES = {
    '0': 'sadness',
    '1': 'joy',
    '2': 'love',
    '3': 'anger',
    '4': 'fear',
    '5': 'surprise',
}

# create a SparkSession
spark = SparkSession.builder.appName("CSV to DataFrame").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()

df = spark.read.format("csv").option("header", "true").load("code/text.csv")



# Find the 10 most common words for each class in the test dataset and save to a csv
def common_words(df):

    # Split sentences into a list of words
    df = df.withColumn("split_text", split("text", "\s+"))

    # filter out stopwords
    remover = StopWordsRemover(inputCol="split_text", outputCol="filtered_words")

    # add some stopwords
    stop_words = ["im", "really", "ive", "feel", "feels", "feeling", "get", "got"]
    remover.setStopWords(remover.getStopWords() + stop_words)

    df = remover.transform(df)

    # Create an empty DataFrame with a schema
    schema = StructType([
        StructField("label", DoubleType(), True),
        StructField("word", StringType(), True),
        StructField("count", DoubleType(), True)
    ])

    return_df = spark.createDataFrame([], schema)

    for label in CLASSES.keys():
        # Filter by class
        class_df = df.filter(df["label"] == str(label))

        # Explode words
        class_df = class_df.withColumn("word", explode(col("filtered_words")))

        # Group by word and count
        word_count = class_df.groupBy("word").count()

        # Order by count
        word_count = word_count.orderBy(desc("count"))

        # Get the top 10 words
        top_words = word_count.limit(10)

        # add column with the class label
        top_words = top_words.withColumn("label", lit(label))

        # reorder columns
        top_words = top_words.select("label", "word", "count")

        return_df = return_df.union(top_words)

    return_df.coalesce(1).write.csv("code/common_words", header=True, mode="overwrite")


common_words(df)