"""
Image Feature Extraction with PySpark and MobileNetV2

This script performs distributed image feature extraction using:
- MobileNetV2 for deep learning features
- PySpark for distributed processing
- PCA for dimensionality reduction
"""

import os
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf


# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# S3 paths
PATH_TEST = "s3://databricks-workspace-aledplsaled-bucket/data/Test"
PATH_TRAIN = "s3://databricks-workspace-aledplsaled-bucket/data/Training"
PATH_RESULT = "s3://databricks-workspace-aledplsaled-bucket/Results"

DATA_PATHS = [PATH_TEST, PATH_TRAIN]

# AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Validate credentials
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError(
        "AWS credentials not found."
    )


# ============================================================================
# SPARK SESSION SETUP
# ============================================================================

def initialize_spark():
    """Initialize and configure Spark session with S3 access."""
    spark = (
        SparkSession.builder
        .appName("ImageFeatureExtraction")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.parquet.writeLegacyFormat", "true")
        .getOrCreate()
    )
    
    # Configure S3 access
    spark.conf.set("fs.s3a.access.key", AWS_ACCESS_KEY)
    spark.conf.set("fs.s3a.secret.key", AWS_SECRET_KEY)
    spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")
    
    return spark


# ============================================================================
# MODEL SETUP
# ============================================================================

def create_feature_extractor():
    """
    Create MobileNetV2 model without top classification layer.
    Returns the second-to-last layer as feature extractor.
    """
    model = MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3)
    )
    
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False
    
    # Extract features from second-to-last layer
    feature_extractor = Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    
    return feature_extractor


# ============================================================================
# IMAGE PREPROCESSING AND FEATURE EXTRACTION
# ============================================================================

def preprocess_image(content):
    """
    Preprocess raw image bytes for model prediction.
    
    Args:
        content: Raw image bytes
        
    Returns:
        Preprocessed image array ready for MobileNetV2
    """
    try:
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        arr = img_to_array(img)
        return preprocess_input(arr)
    except Exception as e:
        # Return zero array for corrupted images
        print(f"Warning: Failed to process image - {e}")
        return np.zeros((224, 224, 3))


def featurize_series(model, content_series):
    """
    Extract features from a pandas Series of raw images.
    
    Args:
        model: MobileNetV2 feature extractor
        content_series: pd.Series of raw image bytes
        
    Returns:
        pd.Series of flattened feature vectors
    """
    input_batch = np.stack(content_series.map(preprocess_image))
    predictions = model.predict(input_batch)
    
    # Flatten multi-dimensional tensors to vectors
    output = [p.flatten() for p in predictions]
    return pd.Series(output)


def create_featurize_udf(broadcast_weights):
    """
    Create a Pandas UDF for distributed feature extraction.
    
    Args:
        broadcast_weights: Broadcasted model weights
        
    Returns:
        Pandas UDF function
    """
    def model_fn():
        """Initialize model with broadcasted weights."""
        model = MobileNetV2(
            weights="imagenet",
            include_top=True,
            input_shape=(224, 224, 3)
        )
        for layer in model.layers:
            layer.trainable = False
        feature_extractor = Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        feature_extractor.set_weights(broadcast_weights.value)
        return feature_extractor
    
    @pandas_udf("array<float>", PandasUDFType.SCALAR_ITER)
    def featurize_udf(content_series_iter):
        """
        Scalar Iterator pandas UDF for batch feature extraction.
        Loads model once and reuses it across batches for efficiency.
        """
        model = model_fn()
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)
    
    return featurize_udf


# ============================================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================================

def perform_pca(features_df, input_column="features", output_column="pca_features"):
    """
    Apply PCA for dimensionality reduction with automatic component selection.
    Selects minimum components to retain 95% of variance.
    
    Args:
        features_df: DataFrame with feature arrays
        input_column: Name of column containing features
        output_column: Name of column for PCA features
        
    Returns:
        DataFrame with PCA-reduced features
    """
    # Convert array to vector for PCA
    @udf(VectorUDT())
    def array_to_vector(array):
        return Vectors.dense(array)
    
    df_with_vectors = features_df.withColumn(
        "feature_vector",
        array_to_vector(input_column)
    )
    
    # Initial PCA with 250 components to find optimal number
    n_components_initial = 250
    pca_initial = PCA(
        k=n_components_initial,
        inputCol="feature_vector",
        outputCol="pca_features_temp"
    )
    pca_model_initial = pca_initial.fit(df_with_vectors)
    
    # Find optimal number of components for 95% explained variance
    cumulative_variance = pca_model_initial.explainedVariance.cumsum()
    optimal_n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Optimal number of components: {optimal_n_components} (95% variance explained)")
    
    # Refit PCA with optimal components
    pca = PCA(
        k=optimal_n_components,
        inputCol="feature_vector",
        outputCol="pca_features_vector"
    )
    pca_model = pca.fit(df_with_vectors)
    df_with_pca = pca_model.transform(df_with_vectors)
    
    # Convert PCA features back to array type
    @udf(ArrayType(FloatType()))
    def vector_to_array(vector):
        return vector.toArray().tolist()
    
    df_final = df_with_pca.withColumn(
        output_column,
        vector_to_array("pca_features_vector")
    )
    
    # Display sample results
    print("\nSample PCA features:")
    df_final.select(output_column).show(2, truncate=False)
    
    return df_final


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("IMAGE FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    
    # Initialize Spark
    print("\n[1/6] Initializing Spark session...")
    spark = initialize_spark()
    
    # Load images
    print(f"\n[2/6] Loading images from {len(DATA_PATHS)} paths...")
    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(DATA_PATHS)
    )
    
    image_count = images.count()
    print(f"Loaded {image_count} images")
    
    # Extract labels from paths
    print("\n[3/6] Extracting labels from paths...")
    images = images.withColumn("label", element_at(split(col("path"), "/"), -2))
    images.select("path", "label").show(5, False)
    
    # Setup model and broadcast weights
    print("\n[4/6] Setting up MobileNetV2 feature extractor...")
    feature_extractor = create_feature_extractor()
    broadcast_weights = spark.sparkContext.broadcast(feature_extractor.get_weights())
    print("Model weights broadcasted to workers")
    
    # Extract features
    print("\n[5/6] Extracting features from images...")
    featurize_udf = create_featurize_udf(broadcast_weights)
    num_partitions = spark.sparkContext.defaultParallelism * 2
    
    features_df = (
        images.repartition(num_partitions)
        .select(
            col("path"),
            col("label"),
            featurize_udf("content").alias("features")
        )
        .cache()
    )
    
    # Apply PCA
    print("\n[6/6] Applying PCA for dimensionality reduction...")
    df_final = perform_pca(features_df, input_column="features", output_column="pca_features")
    
    # Save results
    print(f"\nSaving results to {PATH_RESULT}...")
    df_final.coalesce(1).write.mode("overwrite").parquet(PATH_RESULT)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return df_final


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    result_df = main()
