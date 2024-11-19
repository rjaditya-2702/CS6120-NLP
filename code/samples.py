from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import rand
JAVA_HOME = '/opt/homebrew/opt/openjdk'
SPARK_HOME = '/opt/homebrew/opt/apache-spark/libexec'

CLASSES = {
    '0': 'sadness',
    '1': 'joy',
    '2': 'love',
    '3': 'anger',
    '4': 'fear',
    '5': 'surprise',
}

# create a SparkSession
spark = SparkSession.builder.appName("CSV to DataFrame").getOrCreate()

# define schema
schema = StructType([
    StructField("index", StringType(), True),
    StructField("text", StringType(), True),
    StructField("label", StringType(), True),
])

# convert from csv to pyspark dataframe
df = spark.read.format("csv").option("header", "true").schema(schema).load("code/text.csv")

# show
df.show()

# shuffle data
df = df.orderBy(rand())

# extract 100 samples from each class and save to new .csv files
for label, class_name in CLASSES.items():
    sample = df.filter(df["label"] == str(label)).limit(100)
    sample.write.csv(f"code/samples/{class_name}", header=True)
