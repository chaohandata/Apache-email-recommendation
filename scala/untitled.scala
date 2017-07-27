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

class LSH(numFeatures:Integer, collectionName:String, query:String, zkhost:String, idField: String, contentField: String, outputCollectionName: String, nNeighbors: Integer) {

/*
var numFeatures = 10000
var collectionName = "lucidfind"
var query = "*:*"
var zkhost = "localhost:9983/lwfusion/3.1.0/solr"
var idField = "hash_id"
var contentField = "body"
var outputCollectionName = "email-recs"
*/

private var indexToHashMapping = new HashMap[Long,String]()

def this(collectionName:String, zkhost:String, idField:String, contentField:String, outputCollectionName:String) = this(10000,collectionName,"*:*",zkhost,idField,contentField,outputCollectionName,10)

def this(numFeatures:Integer, collectionName:String, zkhost:String, idField:String, contentField:String, outputCollectionName:String) = this(numFeatures,collectionName,"*:*",zkhost,idField,contentField,outputCollectionName,10)

def this(collectionName:String, query:String, zkhost:String, idField:String, contentField:String, outputCollectionName:String) = this(10000,collectionName,query,zkhost,idField,contentField,outputCollectionName,10)

def this(collectionName:String, zkhost:String, idField:String, contentField:String, outputCollectionName:String, nNeighbors:Integer) = this(10000,collectionName,"*:*",zkhost,idField,contentField,outputCollectionName,nNeighbors)

def this(collectionName:String, query:String, zkhost:String, idField:String, contentField:String, outputCollectionName:String, nNeighbors:Integer) = this(10000,collectionName,query,zkhost,idField,contentField,outputCollectionName,nNeighbors)

def this(numFeatures:Integer, collectionName:String, zkhost:String, idField:String, contentField:String, outputCollectionName:String, nNeighbors:Integer) = this(numFeatures,collectionName,"*:*",zkhost,idField,contentField,outputCollectionName,nNeighbors)

private def convertIndexToHash(row:Row):Array[String] = {
var array:Array[String] = Array[String]()
for(i <- 0 to row.length-1) {
array = array :+ indexToHashMapping(row.getAs[Long](i))
}
array
}

def run() = {
/* 
Load the data into spark df from solr, clean the data remove missing datapoints, remove outliers.
*/
val sqlContext = spark.sqlContext
val options = Map(
"collection" -> collectionName,
"query"->query,
"zkhost" -> zkhost
)

//remove missing data points
var rawData = spark.read.format("solr").options(options).load().select(idField,contentField)
var tempDF = rawData.rdd.filter{case c=> !c.isNullAt(1)}.map{case c => (c.getAs[String](0),c.getAs[String](1),c.getAs[String](1).length)}.toDF("hash_id","contents","length")
// Calculate statistics to remove the outliers.
var avgLength: Double = tempDF.agg(avg("length")).rdd.map{case c=>c.getDouble(0)}.collect()(0)
var std: Double = tempDF.agg(stddev_samp("length")).rdd.map{case c=> c.getDouble(0)}.collect()(0)
val limit: Double = avgLength+3*std
var df = tempDF.rdd.filter{case c=> !(c.getInt(2) > limit)}.map{case c=> (c.getString(0),c.getString(1))}.toDF("hash_id":String,"contents":String)

/*
Stem, tokenize, and Stop Words removal
*/
val stemmer = new Stemmer().setInputCol("contents").setOutputCol("stemmed").setLanguage("English")
val stemmed = stemmer.transform(df)
stemmed.show
val regexTokenizer = new RegexTokenizer().setInputCol("stemmed").setOutputCol("tokens").setPattern("[^a-zA-Z]+").setToLowercase(true)
val regexTokenized = regexTokenizer.transform(stemmed)
regexTokenized.show
val stopWordsRemover = new StopWordsRemover("stopWords").setInputCol("tokens").setOutputCol("stopwords")
val stopWordsRemoved = stopWordsRemover.transform(regexTokenized)
stopWordsRemoved.show

/*
Use hashingTF and IDF to vectorize the text data (contents column)
*/
val hashingTF = new HashingTF().setInputCol("stopwords").setOutputCol("rawFeatures").setNumFeatures(numFeatures)
val featurizedData = hashingTF.transform(stopWordsRemoved)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val rescaledData = idfModel.transform(featurizedData)
rescaledData.select("hash_id", "features").show()

/* 
Bring the data in the format of mllib.SparseVectors to calculate the nearest neighbors 
*/
var rddVector = rescaledData.select("hash_id","features").rdd.map{case row => (
row.getAs[String]("hash_id"),
org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
)}.zipWithIndex
// index to unique-id mapping for reconstructing the recommendations
var indexToHashMapping = new HashMap[Long,String]()
rddVector.collect.foreach{case ((hash,vector),index) => indexToHashMapping+=(index->hash)}
//toSparse converts Vector to SparseVector
val indexedRDD = rddVector.map{ case((hash,vector), index) => (index, vector.toSparse)}

/*
Compute the nearest negihbors using the ANN model 
*/
val annModel = new ANN(dimensions = numFeatures,measure = "cosine").setTables(4).setSignatureLength(8).train(indexedRDD)
val neighbors = annModel.neighbors(nNeighbors)
var tempCountDF = neighbors.map{case c=> (c._1,c._2.length)}.toDF
tempCountDF.count
tempCountDF.select("_2").distinct.show

/*
Convert the recomendations from indexes to unique ids.
*/
var recsDF = neighbors.map{case c=> (c._1,c._2.unzip)}.map{case (id,recs)=>(id,recs._1.toList)}.toDF
var nElements = nNeighbors
var outputIndexes = recsDF.select(($"_1" +: Range(0, nElements).map(idx => $"_2"(idx) as "Col" + (idx + 2)):_*))

/*
Save the recommendations to the solr collection
*/
var outputHash = outputIndexes.rdd.map{case c => convertIndexToHash(c)}.map{case c=> c}.toDF("_1")
var outputDF = outputHash.toDF.select((Range(0, nElements+1).map(idx => $"_1"(idx) as "Col" + (idx)):_*))
//outputDF.write.format("solr").options(Map("collection" -> outputCollectionName, "commit_within"->"5000")).save()
outputDF.collect
outputDF.count

println("Job completed")
}
}