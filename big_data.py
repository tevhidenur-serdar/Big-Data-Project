# Import necessary libraries
import os
import zipfile
import shutil
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, length, trim, when
from pyspark.ml import Pipeline
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark import StorageLevel
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import sys

# Path configurations
base_dir = os.getcwd() 
jar_path = os.path.join(base_dir, "spark-nlp-assembly-5.5.1.jar")

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64" 
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Define the path to the TSV file or zip file
zip_file = "amazon_reviews_us_Books_v1_02.tsv.zip"
tsv_file = "amazon_reviews_us_Books_v1_02.tsv"

if os.path.exists(tsv_file):
    print(f"{tsv_file} is already present. No need to extract.")
else:
    if os.path.exists(zip_file):
        print(f"{zip_file} is being extracted... Please wait.")
        start_zip = time.time()
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Extraction completed. Time: {time.time() - start_zip:.2f} seconds")
    else:
        print(f"ERROR: Neither {tsv_file} nor {zip_file} found!")
        sys.exit(1)

# If there is old session, stop it
try:
    spark.stop()
except:
    pass

print("Starting Spark Session...")

# Set Spark memory from environment variable or default to 16g
spark_mem = os.environ.get("SPARK_MEM", "16g")

# Start Spark session with Spark NLP
spark = SparkSession.builder \
    .appName("AmazonBERT") \
    .master("local[*]") \
    .config("spark.driver.memory", spark_mem) \
    .config("spark.executor.memory", spark_mem) \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.jars", jar_path) \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "1000M") \
    .config("spark.jsl.settings.pretrained.project", "public") \
    .getOrCreate()

print("Spark Session started.")

# Data loading and processing

print("Reading and processing data...")

start_time = time.time()

df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", "\t") \
    .csv(tsv_file)

# Sample a small fraction of the data for testing
#df = df.sample(fraction=0.01, seed=42)

# A) Null removal
df_clean = df.dropna(subset=["review_body", "star_rating"])
df_clean = df_clean.withColumn("clean_text", lower(col("review_body"))) \
                   .withColumn("clean_text", regexp_replace("clean_text", "<br />", " ")) \
                   .withColumn("clean_text", regexp_replace("clean_text", "[^a-zA-Z\\s!.?]", "")) \
                   .withColumn("clean_text", trim(col("clean_text")))

# B) Spoiler and short review removal
spoiler_regex = "|".join(["five star", "5 star", "4 star", "1 star"]) # Kısaltıldı
df_filtered = df_clean.filter(~col("clean_text").rlike(spoiler_regex)) \
                     .filter(length(col("clean_text")) > 20)
df_final = df_filtered.withColumn("label", col("star_rating").cast("string")) \
                      .select("clean_text", "label")

# C) Undersampling for class balance
label_counts = df_final.groupBy("label").count().collect()
count_dict = {row["label"]: row["count"] for row in label_counts}
min_count = min(count_dict.values())
fractions = {label: min_count / count for label, count in count_dict.items()}
# Apply Stratified Undersampling
df_final = df_final.sampleBy("label", fractions, seed=42)

# Repartition and persist the final DataFrame
df_final = df_final.repartition(64)
df_final.persist(StorageLevel.MEMORY_AND_DISK)

print(f"Preprocessing is done. Time: {time.time() - start_time:.2f} s")

# BERT model pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("clean_text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

bert_embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L8_512", "en") \
    .setInputCols(["sentence"]) \
    .setOutputCol("sentence_embeddings")

classsifier_dl = ClassifierDLApproach()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("prediction")\
    .setLabelColumn("label")\
    .setMaxEpochs(1) \
    .setLr(0.001) \
    .setBatchSize(32) \
    .setEnableOutputLogs(True)

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    bert_embeddings,
    classsifier_dl
])

# Split the data and train the model
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

print("Training the model...")
model = nlp_pipeline.fit(train_data)

# Save the model
model_save_path = "amazon_bert_model_portable"
model.write().overwrite().save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the model
print("Evaluating the model...")
predictions = model.transform(test_data)
preds_df = predictions.select("label", "prediction.result").toPandas()
preds_df['result'] = preds_df['result'].apply(lambda x: x[0])

print("Classification Report:")
print(classification_report(preds_df['label'], preds_df['result']))
