package image

/**
 * Created by xiaochen.tian on 10/1/2015.
 */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils


object HandingWriting2 extends App{

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

  val path = "file:////Users/xiaochen.tian/Google Drive/work/handwriting/training.svmfile"
  val tpath = "file:////Users/xiaochen.tian/Google Drive/work/handwriting/testing.svmfile"


  def main = {


    // Load training data
    val train = MLUtils.loadLibSVMFile(sc, path).toDF()
    val test = MLUtils.loadLibSVMFile(sc, tpath).toDF()

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

  }


  //sc.textFile(path).take(100) foreach println

  main



}
