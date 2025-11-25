# OC_Projet_9
Project on Big data, cloud computing and PySpark.

Workflow:
- Data stored in s3
- Compute unit in Databricks (i3.2xlarge - 61GB RAM)
- Notebook in Databricks
- Pyspark to process the data

Data: 
- Fruit dataset

Goal: 
- Extract feature from the fruits images with the PCA algoritm and store them back to s3

1. Data stored on s3
2. Initialize the Spark session
3. Initialize MobileNetV2 with 'imagenet' weights to for feature extraction
4. Load the images in batches on the cluster with Spark
5. Extract the features
6. Run PCA to reduce the dimension and keep 95% of the variance
7. Save results back to the s3 bucket
