name := "sparkmllib"

version := "1.0"

scalaVersion := "2.11.7"


fork := true

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "1.5.0",
  "org.apache.spark" % "spark-graphx_2.11" % "1.5.0",
  "org.apache.spark" % "spark-sql_2.11" % "1.5.0",
  "org.apache.spark" % "spark-mllib_2.11" % "1.5.1",
  "com.databricks" % "spark-csv_2.11" % "1.1.0"
)