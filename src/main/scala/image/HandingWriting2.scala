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

  implicit val sc = new SparkContext("local[3]","appName")

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  val training_path = ""
  val testing_path = ""

  def disable_log = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
  }

  disable_log

  //val path = "file:////Users/xiaochen.tian/gd/work/handwriting/training.svmfile"
  //val tpath = "file:////Users/xiaochen.tian/gd/work/handwriting/testing.svmfile"

  val path = "file:////Users/xiaochen.tian/gd/work/handwriting/chinese_train_handwriting.svmfile"
  val tpath = "file:////Users/xiaochen.tian/gd/work/handwriting/chinese_test_handwriting.svmfile"


  def main = {


    // Load training data
    val train = MLUtils.loadLibSVMFile(sc, path).toDF()

    val test = MLUtils.loadLibSVMFile(sc, tpath).toDF()

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
    //val layers = Array[Int](784, 20,30, 100)
    val layers = Array[Int](400, 20,30, 100)
    //val layers = Array[Int](784, 20,20, 10)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(12345L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")


    //result.rdd.zipWithIndex().filter { x=> x._1(0) != x._1(2) } collect() map( x => x._1(2) -> x._2 ) foreach println
    println("Precision:" + evaluator.evaluate(predictionAndLabels))

  }


  //sc.textFile(path).take(100) foreach println

  main



}
