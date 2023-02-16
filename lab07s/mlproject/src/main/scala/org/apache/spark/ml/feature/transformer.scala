package org.apache.spark.ml.feature

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, functions => F}

class Url2DomainTransformer(override val uid: String) extends Transformer
  with DefaultParamsWritable
{
  def this() = this(Identifiable.randomUID("Url2DomainTransformer"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn("host", F.lower(F.callUDF("parse_url", col("url"), F.lit("HOST"))))
      .withColumn("domain", F.regexp_replace(col("host"), "www.", ""))
      .drop("host")
  }

  override def copy(extra: ParamMap): Url2DomainTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val idx = schema.fieldIndex("url")
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    schema.add(StructField("domain", StringType, false))
  }

}

object Url2DomainTransformer extends DefaultParamsReadable[Url2DomainTransformer] {
  override def load(path: String): Url2DomainTransformer = super.load(path)
}