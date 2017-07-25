import com.lucidworks.spark.rdd.SolrRDD
import com.github.karlhigley.spark.neighbors.ANN
import org.apache.spark.rdd.RDD
import java.io._
import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF, Tokenizer, CountVectorizer, CountVectorizerModel, StopWordsRemover}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.distributed._
import collection.mutable.HashMap
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{stddev_samp, stddev_pop}

val sqlContext = spark.sqlContext
//val solrRDD = SolrRDD("localhost:9983/lwfusion/3.1.0/solr", "lucidfind", sc)
val options = Map(
  "collection" -> "lucidfind",
  "zkhost" -> "localhost:9983/lwfusion/3.1.0/solr"
)
val rawDF = spark.read.format("solr").options(options).load
var tempRDD = rawDF.rdd.map{case c=> (c.getAs[String](15),c.getAs[String](31) + "\n" + c.getAs[String](7))}
var tempDF = tempRDD.map{case c=> (c._1,c._2,c._2.length)}.toDF("hash_id","contents","length")

var avgLength: Double = tempDF.agg(avg("length")).rdd.map{case c=>c.getDouble(0)}.collect()(0)
var std: Double = tempDF.agg(stddev_samp("length")).rdd.map{case c=> c.getDouble(0)}.collect()(0)
val limit: Double = avgLength+3*std

var df = tempDF.rdd.filter{case c=> !(c.getInt(2) > limit)}.map{case c=> (c.getString(0),c.getString(1))}.toDF("hash_id":String,"contents":String)
df.show

val stemmer = new Stemmer().setInputCol("contents").setOutputCol("stemmed").setLanguage("English")
val stemmed = stemmer.transform(df)
stemmed.show
val regexTokenizer = new RegexTokenizer().setInputCol("stemmed").setOutputCol("tokens").setPattern("[^a-zA-Z]+").setToLowercase(true)
val regexTokenized = regexTokenizer.transform(stemmed)
regexTokenized.show
val stopWordsRemover = new StopWordsRemover("stopWords").setInputCol("tokens").setOutputCol("stopwords")
val stopWordsRemoved = stopWordsRemover.transform(regexTokenized)
stopWordsRemoved.show

val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("stopwords").setOutputCol("features").setMinDF(2).fit(stopWordsRemoved)
var rescaledData = cvModel.transform(stopWordsRemoved)

var rddVector = rescaledData.select("hash_id","features").rdd.map{case row => (
   row.getAs[String]("hash_id"),
   org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
)}.zipWithIndex

//var hashToIndexMapping = new HashMap[String,Long]()
//rddVector.collect.foreach{case ((hash,vector),index) => hashToIndexMapping+=(hash->index)}

val indexedRDD = rddVector.map{
    case((hash,vector), index) => (index, vector.toSparse)
}

val annModel = new ANN(measure = "cosine").setTables(4).setSignatureLength(64).train(indexedRDD)
val neighbors = annModel.neighbors(5)






