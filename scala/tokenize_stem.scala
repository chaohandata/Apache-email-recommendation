// Code to do clustering on documents.
import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.distributed._
import collection.mutable.HashMap

val sentenceData = spark.createDataFrame(Seq(
  ("hash4", "This needs a lot of cleaning and classes in Spark"),
  ("hash5", "Logistic regression is widely used algorithm"),
  ("hash1", "Hi I heard about Spark123 D##### ssps$.`WA!"),
  ("hash2", "I wish Java could use case classes"),
  ("hash3", "Logistic regression models are neat!@#!$dvfs")
)).toDF("label", "sentence")

val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("[^a-zA-Z]+")
val regexTokenized = regexTokenizer.transform(sentenceData)

val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(3).setMinDF(2).fit(regexTokenized)
var rescaledData = cvModel.transform(regexTokenized)
//val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20000)
//val featurizedData = hashingTF.transform(regexTokenized)

//val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//val idfModel = idf.fit(featurizedData)
//val rescaledData = idfModel.transform(featurizedData)

// CLUSTERING DOCUMENTS
val numClusters = 2
val numIterations = 20
var kmeans = new KMeans().setK(2).setSeed(1L).setFeaturesCol("features").setPredictionCol("prediction")
var model = kmeans.fit(rescaledData)
var newDF = kmeans.transform(rescaledData)
val clusters = KMeans.train(rescaledData.select("features"), numClusters, numIterations)

var rddVector = rescaledData.select("label","features").rdd.map{case row => (
   row.getAs[String]("label"),
   org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
)}.zipWithIndex

var hashToIndexMapping = new HashMap[String,Long]()
rddVector.collect.foreach{case ((hash,vector),index) => hashToIndexMapping+=(hash->index)}

val indexedRDD = rddVector.map{
    case((hash,vector), index) => IndexedRow(index, vector)
}
//make a matrix 
val matrix = new IndexedRowMatrix(indexedRDD)
//calculate the distributions
val dist = matrix.toCoordinateMatrix.transpose().toIndexedRowMatrix().columnSimilarities()

dist.numCols
dist.numRows
dist.entries.collect








