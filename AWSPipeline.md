### Feature Enginnering Code

```py 
# Import some modules we will need later on
sc.setLogLevel("ERROR")
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler from pyspark.ml 
import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up the path to a group of files. Should match 'Home', 'Home Improvement' and 'Home Entertainment' bucket = 's3a://amazon-reviews-pds/tsv/'
filename = 'amazon_reviews_us_Digital*_v1_00.tsv.gz'
file_path = bucket + filename

# Create a Spark Dataframe from some Amazon reviews data
sdf = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True

# Get a random sample of the data. 25% sampled
sdf = sdf.sample(.25)


# Drop some of the records where the certain columns are empty (null or na)
sdf = sdf.na.drop(subset=["star_rating", "review_body", "review_date", "vine", "product_category", "verified_purchase"])

sdf.groupby("product_category").count().show(truncate=False)
```
<img width="513" alt="image" src="https://user-images.githubusercontent.com/92469431/208258868-ea47ae6f-7c02-4a94-8983-ded0c23b5832.png">


```py
# Drop any records that are not in our three target product categories.
sdf = sdf.filter(sdf.product_category.isin(['Digital_Video_Games','Digital_Video_Download','Digital_Software', 'Digital_Music_Purchase','Digital_Ebook_Purchase']))

# Create a count of the review words
sdf = sdf.withColumn('review_body_wordcount', size(split(col('review_body'), ' ')))
# Create a binary output label
sdf = sdf.withColumn("label", when(col("star_rating") > 3, 1.0).otherwise(0.0))

# Cast the total_votes and review_body_wordcount to double
sdf = sdf.withColumn("total_votes",sdf.total_votes.cast(DoubleType()))
sdf = sdf.withColumn("review_body_wordcount",sdf.review_body_wordcount.cast(DoubleType()))

# Drop some columns that we no longer need
sdf = sdf.drop("review_body", "review_headline", "marketplace", "customer_id", "review_id", "product_id", "product_parent", "product_title")

# Split the data into training and test sets
trainingData, testData = sdf.randomSplit([0.7, 0.3], seed=3456)

# Create an indexer for the three string based columns.
indexer = StringIndexer(inputCols=["product_category", "vine", "verified_purchase"], outputCols=["product_categoryIndex", "vineIndex",
"verified_purchaseIndex"], handleInvalid="keep")

# Create an encoder for the three indexes and the age integer column.
encoder = OneHotEncoder(inputCols=["product_categoryIndex", "vineIndex", "verified_purchaseIndex" ],
outputCols=["product_categoryVector", "vineVector", "verified_purchaseVector" ], dropLast=True, handleInvalid="keep")

# Create an assembler for the individual feature vectors and the float/double columns
assembler = VectorAssembler(inputCols=["product_categoryVector", "vineVector", "verified_purchaseVector", "total_votes",
"review_body_wordcount"], outputCol="features")

# Create a LogisticRegression Estimator
lr = LogisticRegression(maxIter=10)
```
### Model Results
```py
# Test the predictions
predictions = cv.transform(testData)

# Calculate AUC
auc = evaluator.evaluate(predictions)
print('AUC:', auc)
```
<img width="513" alt="image" src="https://user-images.githubusercontent.com/92469431/208259816-331ad96f-688d-45c7-8d13-903423bef52f.png">

```py
# Create the confusion matrix 
predictions.groupby('label').pivot('prediction').count().fillna(0).show()
cm = predictions.groupby('label').pivot('prediction').count().fillna(0).collect()
def calculate_precision_recall(cm): 
  tn = cm[0][1]
  fp = cm[0][2]
  fn = cm[1][1]
  tp = cm[1][2]
  precision = tp / ( tp + fp )
  recall = tp / ( tp + fn )
  accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
  f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) ) return accuracy, precision, recall, f1_score
print( calculate_precision_recall(cm) )
```
<img width="513" alt="image" src="https://user-images.githubusercontent.com/92469431/208260321-90a9c24c-38eb-4fb4-81c0-4e12f372eb4b.png">

### Parameters and ROC Curve

```py
parammap = cv.bestModel.stages[3].extractParamMap()
for p, v in parammap.items():
    print(p, v)
    
# Grab the model from Stage 3 of the pipeline 
mymodel = cv.bestModel.stages[3]
import matplotlib.pyplot as plt 
plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'r--')
x = mymodel.summary.roc.select('FPR').collect() 
y = mymodel.summary.roc.select('TPR').collect() plt.scatter(x, y)
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title("ROC Curve") 
plt.savefig("reviews_roc.png")
```
![image](https://user-images.githubusercontent.com/92469431/208261021-d31be776-206b-4b47-99e3-bc2ef1d00c5e.jpeg)
![image](https://user-images.githubusercontent.com/92469431/208261057-250f0360-db71-4fdf-a21c-e4b00f625576.jpeg)

### Coefficients
```py
# Extract the coefficients on each of the variables 
coeff = mymodel.coefficients.toArray().tolist()

#Loop thorugh the geatures to extract the original column names. Store in the var_index dictionary
var_index = dict ()
for variable_type in ['numeric', 'binary']:
    for variable in predictions.schema["features"].metadata["ml_attr"]["attrs"][variable_type]:
        print("Found variable:", variable)
        idx = variable['idx']
        name = variable['name']
        var_index[idx] = name
#Loop thorugh all the variables found and print out the associated coefficients
for i in range(len(var_index)):
    print(i, var_index[i], coeff[i])
```
<img width="513" alt="image" src="https://user-images.githubusercontent.com/92469431/208261896-ff37b034-b1da-4a75-90b6-34a65b7df1fa.png">
