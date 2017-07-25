// Sanket Shahane
// Analyzing the emails data to see some aggregation results.
// Read data from solr, print it. 

// not a good idea as the data is not loaded in a structured format
import com.lucidworks.spark.rdd.SolrRDD
val solrRDD = SolrRDD("localhost:9983/lwfusion/3.1.0/solr", "lucidfind", sc)

// load it as a dataframe and 
import org.apache.spark.rdd.RDD
import java.io._

val sqlContext = spark.sqlContext
val options = Map(
  "collection" -> "lucidfind",
  "zkhost" -> "localhost:9983/lwfusion/3.1.0/solr"
)
val df = spark.read.format("solr").options(options).load

df.count
var rdd = df.rdd.map{case c => (c(15),c(31),c(7))}
rdd.foreach{case y=>
val pw = new PrintWriter(new File("/Users/sanket/Desktop/nlp_emailrecs/sample_data/"+y._1+".email"))
pw.write(y._2+"\n"+y._3)
pw.close
}

rdd.count
var rdd1 = rdd.map{case c => (c(15),c(31),c(7))}
rdd1.count

df.count
df.select("datasource_label").groupBy("datasource_label").count
df.select("subject").groupBy("subject").count.orderBy(desc("count")).show
df.select("from_email").distinct().count
df.select("subject").distinct().count

var documents = df.select("hash_id","subject","body").rdd.map{case c=>(c(0).toString,c(1).toString,c(2).toString)}

df.select("hash_id").count

var x = documents.take(1)
documents.foreach{ case y=>
val pw = new PrintWriter(new File("/Users/sanket/Desktop/nlp_emailrecs/sample_data/"+y._1+".email"))
pw.write(y._2+"\n"+y._3)
pw.close
}
/* Python code to tokenize only words 

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')

/*

