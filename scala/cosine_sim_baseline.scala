import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.distributed._
import collection.mutable.HashMap

val sentenceData = spark.createDataFrame(Seq(
  ("hash4", "Logistic regression not so nice algorithm"),
  ("hash5", "Logistic regression is widely used algorithm"),
  ("hash1", "Logistic regression is widely used algorithm"),
  ("hash2", "Logistic regression is widely used algorithm"),
  ("hash3", "Logistic regression kind of good algorithm but difficult to converge")
)).toDF("label", "sentence")

val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("[^a-zA-Z]+")
val regexTokenized = regexTokenizer.transform(sentenceData)

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(3).setMinDF(2).fit(regexTokenized)
var rescaledData = cvModel.transform(regexTokenized)

//val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(25)
//val featurizedData = hashingTF.transform(regexTokenized)

//val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//val idfModel = idf.fit(featurizedData)
//val rescaledData = idfModel.transform(featurizedData)

var rddVector = rescaledData.select("label","features").rdd.map{case row => (
   row.getAs[String]("label"),
   org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
)}.zipWithIndex

var hashToIndexMapping = new HashMap[String,Long]()
rddVector.collect.foreach{case ((hash,vector),index) => hashToIndexMapping+=(hash->index)}

// Above code works fine

val indexedRDD = rddVector.map{
    case((hash,vector), index) => IndexedRow(index, vector)
}
val matrix = new IndexedRowMatrix(indexedRDD)
//calculate the distributions
val dist = matrix.toCoordinateMatrix.transpose().toIndexedRowMatrix().columnSimilarities()

dist.entries.collect.foreach(println)

```
scala> dist.numCols
res1: Long = 5

scala> dist.numRows
res2: Long = 5

scala> dist.entries.collect
res3: Array[org.apache.spark.mllib.linalg.distributed.MatrixEntry] = Array(MatrixEntry(1,4,0.16599680709692669), MatrixEntry(2,3,0.06342509090459066), MatrixEntry(0,2,0.05148545147809198), MatrixEntry(0,3,0.05575017364415862))
```
I expect to get a complete matrix like 5*5 or at least the upper triangle entries of the similarity matrix. But for some reason it does not give it. Any idea what is the problem here?

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
# cosine_sim is a nDoc*nDoc similarity matrix I choose the to 10
x=cosine_sim[0]
x.argsort()[::-1][:10]

I simply wanna do this ^ but in Spark-Scala