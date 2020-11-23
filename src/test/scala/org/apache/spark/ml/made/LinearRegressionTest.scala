package org.apache.spark.ml.made

import breeze.linalg.DenseVector
import breeze.numerics.round
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row, functions}
import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.0001
  val weights_inaccuracy = 0.1

  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors

  lazy val rand_data: DataFrame = LinearRegressionTest._rand_data
  lazy val rand_points: Seq[Vector] = LinearRegressionTest._rand_points

  "Model" should "calculate a linear combination from the data" in {
    val weights: Vector = Vectors.dense(1.1, -2.0, 1.7)
    val bias: Double = 1.3

    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features")
      .setOutputCol("prediction")

    val values = model.transform(data).collect().map(_.getAs[Double](1))

    values.length should be(2)

    values(0) should be(vectors(0)(0) * weights(0) + vectors(0)(1) * weights(1) + vectors(0)(2) * weights(2) + bias +- delta)
    values(1) should be(vectors(1)(0) * weights(0) + vectors(1)(1) * weights(1) + vectors(1)(2) * weights(2) + bias +- delta)
  }

  "Estimator" should "produce functional model" in {
    val weights: Vector = Vectors.dense(1.1, -2.0, 1.7)
    val bias: Double = 1.3
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features")
      .setOutputCol("label")

    val train = true_model
      .transform(rand_data)
      .select(functions.col("features"), (functions.col("label") + functions.rand() * functions.lit(0.1) - functions.lit(0.05)).as("label"))

    val estimator: LinearRegression = new LinearRegression(1, 1000)
      .setInputCol("features")
      .setOutputCol("label")
    val model = estimator.fit(train)

    model.weights(0) should be(weights(0) +- weights_inaccuracy)
    model.weights(1) should be(weights(1) +- weights_inaccuracy)
    model.weights(2) should be(weights(2) +- weights_inaccuracy)
    model.bias should be(bias +- weights_inaccuracy)
  }

  "Estimator" should "work after re-read" in {
    val weights: Vector = Vectors.dense(1.1, -2.0, 1.7)
    val bias: Double = 1.3
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features_test")
      .setOutputCol("label_test")

    val train = true_model
      .transform(rand_data.select(functions.col("features").as("features_test")))
      .select(
        functions.col("features_test"),
        (functions.col("label_test") + functions.rand() * functions.lit(0.1) - functions.lit(0.05)).as("label_test")
      )

    val stepSize = 1.0
    val numIterations = 1000
    var pipeline = new Pipeline().setStages(Array(
      new LinearRegression(stepSize, numIterations)
        .setInputCol("features_test")
        .setOutputCol("label_test")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    pipeline = Pipeline.load(tmpFolder.getAbsolutePath)
    pipeline.getStages(0).asInstanceOf[LinearRegression].stepSize should be(stepSize)
    pipeline.getStages(0).asInstanceOf[LinearRegression].numIterations should be(numIterations)

    val model = pipeline.fit(train).stages(0).asInstanceOf[LinearRegressionModel]

    model.weights(0) should be(weights(0) +- weights_inaccuracy)
    model.weights(1) should be(weights(1) +- weights_inaccuracy)
    model.weights(2) should be(weights(2) +- weights_inaccuracy)
    model.bias should be(bias +- weights_inaccuracy)
  }

  "Model" should "work after re-read" in {
    val weights: Vector = Vectors.dense(1.1, -2.0, 1.7)
    val bias: Double = 1.3
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features_test")
      .setOutputCol("label_test")

    val train = true_model
      .transform(rand_data.select(functions.col("features").as("features_test")))
      .select(
        functions.col("features_test"),
        (functions.col("label_test") + functions.rand() * functions.lit(0.1) - functions.lit(0.05)).as("label_test")
      )

    val stepSize = 1.0
    val numIterations = 1000
    var pipeline = new Pipeline().setStages(Array(
      new LinearRegression(stepSize, numIterations)
        .setInputCol("features_test")
        .setOutputCol("label_test")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.fit(train).write.overwrite().save(tmpFolder.getAbsolutePath)

    val pipeline_model = PipelineModel.load(tmpFolder.getAbsolutePath)
    val model = pipeline_model.stages(0).asInstanceOf[LinearRegressionModel]

    model.getInputCol should be("features_test")
    model.getOutputCol should be("label_test")
    model.weights(0) should be(weights(0) +- weights_inaccuracy)
    model.weights(1) should be(weights(1) +- weights_inaccuracy)
    model.weights(2) should be(weights(2) +- weights_inaccuracy)
    model.bias should be(bias +- weights_inaccuracy)
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(13.5, 12, 7.0),
    Vectors.dense(-1, 0, 3.2),
  )
  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _rand_points = Seq.fill(30)(Vectors.fromBreeze(DenseVector.rand(3)))
  lazy val _rand_data: DataFrame = {
    import sqlc.implicits._
    _rand_points.map(x => Tuple1(x)).toDF("features")
  }
}