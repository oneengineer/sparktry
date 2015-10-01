package image

/**
 * Created by xiaochen.tian on 10/1/2015.
 */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row

import org.apache.spark.rdd._


object HandingWriting extends App{

  implicit val sc = new SparkContext("local[1]","appName")

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  val training_path = ""
  val testing_path = ""

  def disable_log = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
  }

  disable_log

  val path = "file:///D:\\vms\\sync\\work\\handwriting\\0.txt"


  def main = {


    // Load training data
    val data = MLUtils.loadLibSVMFile(sc, path).toDF()
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))


    result.rdd.zipWithIndex().filter { x=> x._1(0) != x._1(2) } collect() map( x => x._1(2) -> x._2 ) foreach println

    //result.select("label","prediction").rdd.zipWithIndex().filter { case (row,id) => row(0).asInstanceOf[Double] != row(1).asInstanceOf[Double] } take(100) foreach println
    //result.rdd.zipWithIndex().filter { case (row,id) => row(0).asInstanceOf[Double] != row(1).asInstanceOf[Double] } take(100) foreach println


    //result.filter( result("label") !== result("prediction") ).show(false)//.select("features","prediction").show(false)

  }


  //sc.textFile(path).take(100) foreach println

  main



}
