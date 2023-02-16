ThisBuild / version := "1.0"

ThisBuild / scalaVersion := "2.11.12"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.7" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.7" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.7" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql-kafka-0-10" % "2.4.7"

lazy val root = (project in file("."))
  .settings(
    name := "mlproject"
  )
