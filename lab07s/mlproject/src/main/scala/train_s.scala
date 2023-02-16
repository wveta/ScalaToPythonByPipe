import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, SklearnEstimator, StringIndexer, Url2DomainTransformer}
import org.apache.spark.sql.{SparkSession, functions => F}


object train {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("lab07s_train")
      .getOrCreate()

    import spark.implicits._

    spark.conf.set("spark.sql.session.timeZone", "UTC")

    val inputPath = spark.sparkContext.getConf.get("spark.mlproject.input_dir")
    val otputPath = spark.sparkContext.getConf.get("spark.mlproject.output_dir")

    println("logs path: " + inputPath)
    println("model path: " + otputPath)

    val inputData = spark.read
      .json(inputPath)
      .select($"uid", $"gender_age", F.explode($"visits") as "visit")
      .withColumn("url", F.lower($"visit.url"))
      .drop("visit")

    val myTransformer = new Url2DomainTransformer()
    val transformedData = myTransformer.transform(inputData).drop("url")

    val trainData = transformedData
      .groupBy("uid")
      .agg(
        F.collect_list("domain") as "domains",
        F.first("gender_age") as "gender_age"
      )

    val cv = new CountVectorizer()
      .setInputCol("domains")
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("gender_age")
      .setOutputCol("label")

    val pyScriptPath = "/data/home/andrey.lykov/lab07/TestScalaPython.py"
    spark.sparkContext.addFile(pyScriptPath)

    val lr = new SklearnEstimator()

    val pipeline = new Pipeline()
      .setStages(Array(cv, indexer, lr))

    val model = pipeline.fit(trainData.limit(1))
    println(model)

//    model.write.overwrite().save(otputPath)

    spark.stop()
  }
}
