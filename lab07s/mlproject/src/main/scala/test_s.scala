import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark
import org.apache.spark.SparkFiles
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SparkSession, functions => F}


//spark-submit --conf spark.mlproject.in_topic=andrey_lykov --conf spark.mlproject.model_dir=/user/andrey.lykov/lab07_model  --conf spark.mlproject.out_topic=andrey_lykov_lab07_out --class test --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.7  ./target/scala-2.11/mlproject_2.11-1.0.jar >logTestLab7.txt

object test {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("lab07_test")
      .getOrCreate()

    import spark.implicits._

//    spark.conf.set("spark.sql.session.timeZone", "UTC")
    val modelPath: String = spark.conf.get("spark.mlproject.model_dir")
//    val modelPath = spark.sparkContext.getConf.get("spark.mlproject.model_dir")
    println("modelPath: " + modelPath)
    val kafkaInput: String = spark.conf.get("spark.mlproject.in_topic")
//    val kafkaInput = spark.sparkContext.getConf.get("spark.mlproject.in_topic")
    println("kafkaInput: " + kafkaInput)
    val kafkaOutput: String = spark.conf.get("spark.mlproject.out_topic")
//    val kafkaOutput = spark.sparkContext.getConf.get("spark.mlproject.out_topic")
    println("kafkaOutput: " + kafkaOutput)

    val model = PipelineModel.load(modelPath)
    val indexer = model.stages(1).asInstanceOf[StringIndexerModel]
    val innerModel = model.stages(2).asInstanceOf[LogisticRegressionModel]

    val checkpointPath = "/user/andrey.lykov/checkpoint_lab07"

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    fs.delete(new Path(checkpointPath), true)

    val inputStream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "spark-master-1:6667")
      .option("subscribe", kafkaInput)
      .load()
    println("inputStream Load")

    val dataSchema = new StructType(
      Array(
        StructField("url", StringType),
        StructField("timestamp", LongType)
      )
    )

    val visitsSchema = new StructType(
      Array(
        StructField("uid", StringType),
        StructField("visits", ArrayType(dataSchema))
      )
    )

    val originalData = inputStream
      .select(F.from_json($"value".cast("string"), visitsSchema) as "value")
      .select($"value.*")

    val domains = originalData
      .select($"uid", F.explode($"visits") as "visit")
      .withColumn("host", F.lower(F.callUDF("parse_url", $"visit.url", F.lit("HOST"))))
      .withColumn("domain", F.regexp_replace($"host", "www.", ""))
      .select($"uid", $"domain")

    val inputForModel = domains
      .groupBy("uid")
      .agg(
        F.collect_list($"domain") as "domains"
      )
    println("data ready")
    val predicts = model.transform(inputForModel)
    println("model ready")
    val converter = new IndexToString()
      .setInputCol(innerModel.getPredictionCol)
      .setOutputCol(indexer.getInputCol)
      .setLabels(indexer.labels)

    val outOfModel = converter
      .transform(predicts)
      .select($"uid", $"gender_age")
      .toJSON
    println("json ready")

    val wrStream = outOfModel.writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "spark-master-1:6667")
      .option("topic", kafkaOutput)
      .option("checkpointLocation", checkpointPath)
      .outputMode("update")

    val startWrite = wrStream.start()

    println("start Write")

    startWrite.awaitTermination()

    println("finish Write")
    spark.sparkContext.addFile()
    spark.stop()
  }
}
