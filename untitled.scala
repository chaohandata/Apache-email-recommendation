// Version 2 code with CountVectorizer Model

import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.distributed._
import collection.mutable.HashMap
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

val sentenceData = spark.createDataFrame(Seq(
  ("hash4", "Logistic regression not so nice algorithm for regression"),
  ("hash5", "Logistic regression is widely used algorithm used algorithm"),
  ("hash1", "Logistic regression is widely used algorithm is widely"),
  ("hash2", "Logistic regression is widely used algorithm Logistic converge converge"),
  ("hash3", "Logistic regression kind of good good algorithm but difficult difficult to converge")
)).toDF("label", "sentence")

val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("[^a-zA-Z]+")
val regexTokenized = regexTokenizer.transform(sentenceData)
val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(3).setMinDF(2).fit(regexTokenized)
var rescaledData = cvModel.transform(regexTokenized)

rescaledData.collect.foreach(println)

var rddVector = rescaledData.select("label","features").rdd.map{case row => (
   row.getAs[String]("label"),
   org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
)}.zipWithIndex

var hashToIndexMapping = new HashMap[String,Long]()
rddVector.collect.foreach{case ((hash,vector),index) => hashToIndexMapping+=(hash->index)}

val indexedRDD = rddVector.map{
    case((hash,vector), index) => IndexedRow(index, vector)
}
val matrix = new IndexedRowMatrix(indexedRDD)
//calculate the distributions
val dist = matrix.toCoordinateMatrix.transpose().toIndexedRowMatrix().columnSimilarities()
dist.entries.collect.foreach(println)
