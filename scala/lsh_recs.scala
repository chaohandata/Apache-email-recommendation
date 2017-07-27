import com.lucidworks.spark.rdd.SolrRDD
import com.github.karlhigley.spark.neighbors.ANN
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF, Tokenizer, CountVectorizer, CountVectorizerModel, StopWordsRemover}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.distributed._
import collection.mutable.HashMap
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{stddev_samp, stddev_pop}
import org.apache.spark.sql.Row

//Some parameters that need to defined
var numFeatures = 10000
var collectionName = "lucidfind"
var zkhost = "localhost:9983/lwfusion/3.1.0/solr"
var idField = "hash_id"
var contentField = "body"
var outputCollectionName = "email-recs"

// Load the data from anywhere and bring in a format of (id,contents)
val sqlContext = spark.sqlContext
val options = Map(
  "collection" -> "lucidfind",
  "zkhost" -> "localhost:9983/lwfusion/3.1.0/solr"
)
val rawDF = spark.read.format("solr").options(options).load
var tempRDD = rawDF.rdd.map{case c=> (c.getAs[String](15),c.getAs[String](31) + "\n" + c.getAs[String](7))}

// Filter out the outliers based on length of the text
var tempDF = tempRDD.map{case c=> (c._1,c._2,c._2.length)}.toDF("hash_id","contents","length")
// Calculate statistics to filter the outliers
var avgLength: Double = tempDF.agg(avg("length")).rdd.map{case c=>c.getDouble(0)}.collect()(0)
var std: Double = tempDF.agg(stddev_samp("length")).rdd.map{case c=> c.getDouble(0)}.collect()(0)
val limit: Double = avgLength+3*std

var df = tempDF.rdd.filter{case c=> !(c.getInt(2) > limit)}.map{case c=> (c.getString(0),c.getString(1))}.toDF("hash_id":String,"contents":String)
df.show
// Stem, tokenize, and Stop Words removal
val stemmer = new Stemmer().setInputCol("contents").setOutputCol("stemmed").setLanguage("English")
val stemmed = stemmer.transform(df)
stemmed.show
val regexTokenizer = new RegexTokenizer().setInputCol("stemmed").setOutputCol("tokens").setPattern("[^a-zA-Z]+").setToLowercase(true)
val regexTokenized = regexTokenizer.transform(stemmed)
regexTokenized.show
val stopWordsRemover = new StopWordsRemover("stopWords").setInputCol("tokens").setOutputCol("stopwords")
val stopWordsRemoved = stopWordsRemover.transform(regexTokenized)
stopWordsRemoved.show


// Need to convert the tokens to features use either CountVectorizer or HashingTF

//val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("stopwords").setOutputCol("features").setMinDF(2).fit(stopWordsRemoved)
//var rescaledData = cvModel.transform(stopWordsRemoved)

val hashingTF = new HashingTF().setInputCol("stopwords").setOutputCol("rawFeatures").setNumFeatures(10000)
val featurizedData = hashingTF.transform(stopWordsRemoved)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val rescaledData = idfModel.transform(featurizedData)
rescaledData.select("hash_id", "features").show()

// Bring the data in the format of mllib.SparseVectors to calculate the nearest neighbors
var rddVector = rescaledData.select("hash_id","features").rdd.map{case row => (
   row.getAs[String]("hash_id"),
   org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
)}.zipWithIndex
// index to id mapping for reconstructing the recommendations
var indexToHashMapping = new HashMap[Long,String]()
rddVector.collect.foreach{case ((hash,vector),index) => indexToHashMapping+=(index->hash)}

val indexedRDD = rddVector.map{
    case((hash,vector), index) => (index, vector.toSparse) //toSparse converts from Vector to SparseVector
}

// Select the number of neighbors we want to compute, pass the parameteres to the ANN model
var nNeighors = 10
val annModel = new ANN(dimensions = 10000,measure = "cosine").setTables(4).setSignatureLength(8).train(indexedRDD)
val neighbors = annModel.neighbors(nNeighors)
var tempCountRDD = neighbors.map{case c=> (c._1,c._2.length)}
var tempCountDF = tempCountRDD.toDF

//tempCountDF.count
//neighbors.take(10).foreach{case c=>println(c._1,c._2.foreach(print))}
// Bringing data back into format that can be written to solr, csv, etc.
var recsDF = neighbors.map{case c=> (c._1,c._2.unzip)}.map{case (id,recs)=>(id,recs._1.toList)}.toDF
var nElements = nNeighors
var outputIndexes = recsDF.select(($"_1" +: Range(0, nElements).map(idx => $"_2"(idx) as "Col" + (idx + 2)):_*))

def convertIndexToHash(row:Row):Array[String] = {
var array:Array[String] = Array[String]()
for(i <- 0 to row.length-1) {
array = array :+ indexToHashMapping(row.getAs[Long](i))
}
array
}

var outputHash = outputIndexes.rdd.map{case c => convertIndexToHash(c)}.map{case c=> c}.toDF("_1")
var outputDF = outputHash.toDF.select((Range(0, nElements+1).map(idx => $"_1"(idx) as "Col" + (idx)):_*))
outputDF.write.csv("/Users/sanket/Desktop/lsh_hash_recs.csv")

