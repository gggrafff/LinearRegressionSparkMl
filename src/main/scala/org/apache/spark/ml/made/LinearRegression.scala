package org.apache.spark.ml.made

import breeze.linalg.sum
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.{Estimator, Model, made}
import org.apache.spark.sql.{DataFrame, Dataset, Row, functions}
import org.apache.spark.sql.types.StructType

/**
 * Характеристика для классов, хранящих информацию о названиях входных и выходных столбцов.
 */
trait HasInOutColumns  extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputCol, "features")
  setDefault(outputCol, "label")
}

/**
 * Характеристика для классов линейной регрессии
 */
trait LinearRegressionParams extends HasInOutColumns {
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkNumericType(schema, getOutputCol)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

/**
 * Эстиматор, порождающий модель линейной регрессии
 * @param uid Идентификатор объекта
 * @param stepSize Learning rate
 * @param numIterations Максимальное количество итераций обучения
 */
class LinearRegression(override val uid: String,
                       val stepSize: Double,
                       val numIterations: Int)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable
    with MLWritable {

  def this() = this(Identifiable.randomUID("linearRegression"), 0.001, 10000)
  def this(uid: String) = this(uid, 0.001, 10000)
  def this(stepSize: Double, numIterations: Int) = this(
    Identifiable.randomUID("linearRegression"),
    stepSize,
    numIterations)

  /**
   * Объект для рассчёта градиентов
   */
  private val gradient = new LeastSquaresGradient()
  /**
   * Объект для обновления весов модели
   */
  private val updater = new SimpleUpdater()
  /**
   * Оптимизатор параметров модели
   */
  private val optimizer = new GradientDescent(gradient, updater, $(inputCol), $(outputCol))
    .setStepSize(stepSize)
    .setNumIterations(numIterations)

  override def setInputCol(value: String) : this.type = {
    set(inputCol, value)
    optimizer.setInputCol($(inputCol))  // Не забываем сообщить оптимизатору об изменении имён столбцов
    this
  }
  override def setOutputCol(value: String): this.type = {
    set(outputCol, value)
    optimizer.setOutputCol($(outputCol))  // Не забываем сообщить оптимизатору об изменении имён столбцов
    this
  }

  /**
   * Обучает модель линейной регрессии
   * @param dataset Тренировочный датасет
   * @return Обученная модель
   */
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    var weights: Vector = Vectors.dense(1.0, 1.0, 1.0, 1.0)  // инициализируем веса модели

    // Добавляем четвёртое измерение для учёта свободного коэффициента линейного закона
    val withOnes = dataset.withColumn("ones", functions.lit(1))
    val assembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "ones"))
      .setOutputCol("features_extended")
    val assembled = assembler
      .transform(withOnes)
      .select(functions.col("features_extended").as($(inputCol)), functions.col($(outputCol)))

    weights = optimizer.optimize(assembled, weights)  // оптимизируем веса

    copyValues(new LinearRegressionModel(new DenseVector(
      weights.toArray.slice(0, weights.size - 1)),
      weights.toArray(weights.size - 1)))
      .setParent(this)  // Возвращаем модель
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = {
    copyValues(new LinearRegression(stepSize, numIterations))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new LinearRegression.LinearRegressionWriter(this)
}

/**
 * Статические члены класса LinearRegression
 */
object LinearRegression extends DefaultParamsReadable[LinearRegression] with MLReadable[LinearRegression] {
  override def read: MLReader[LinearRegression] = new LinearRegressionReader

  override def load(path: String): LinearRegression = super.load(path)

  private class LinearRegressionWriter(instance: LinearRegression) extends MLWriter {

    private case class Data(stepSize: Double, numIterations: Int)

    override protected def saveImpl(path: String): Unit = {
      // Сохраняем метаданные и параметры
      DefaultParamsWriter.saveMetadata(instance, path, sc)

      // Сохраняем другие данные
      val stepSize = instance.stepSize
      val numIterations = instance.numIterations
      val data = Data(stepSize, numIterations)
      val dataPath = new Path(path, "data").toString
      sparkSession
        .createDataFrame(Seq(data))
        .repartition(1)
        .write
        .parquet(dataPath)
    }
  }

  private class LinearRegressionReader extends MLReader[LinearRegression] {
    private val className = classOf[LinearRegression].getName

    override def load(path: String): LinearRegression = {
      // Загружаем метаданные и параметры
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      // Загружаем другие данные
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("stepSize", "numIterations").head()
      val stepSize = data.getAs[Double](0)
      val numIterations = data.getAs[Int](1)

      // Создаём эстиматор с загруженными параметрами
      val transformer = new LinearRegression(metadata.uid, stepSize, numIterations)
      metadata.getAndSetParams(transformer)
      transformer.optimizer.setInputCol(transformer.getInputCol)
      transformer.optimizer.setOutputCol(transformer.getOutputCol)

      transformer
    }
  }
}

/**
 * Класс модели линейной регрессии
 * @param uid Идентификатор объекта модели
 * @param weights Веса модели
 * @param bias Свободный коэффициент линейного закона
 */
class LinearRegressionModel private[made](
                           override val uid: String,
                           val weights: DenseVector,
                           val bias: Double)
  extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(weights, bias))
  }

  /**
   * Получаем прогнозы с помощью модели
   * @param dataset Датасет
   * @return Исходный датасет с прогнозами
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        sum(x.asBreeze *:* weights.asBreeze) + bias
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new LinearRegressionModel.LinearRegressionModelWriter(this)
}

/**
 * Статические члены класса LinearRegressionModel
 */
object LinearRegressionModel extends DefaultParamsReadable[LinearRegressionModel] with MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new LinearRegressionModelReader

  override def load(path: String): LinearRegressionModel = super.load(path)

  private class LinearRegressionModelWriter(instance: LinearRegressionModel) extends MLWriter {

    private case class Data(weights: DenseVector, bias: Double)

    override protected def saveImpl(path: String): Unit = {
      // Сохраняем метаданные и параметры
      DefaultParamsWriter.saveMetadata(instance, path, sc)

      // Сохраняем другие данные
      val weights = instance.weights
      val bias = instance.bias
      val data = Data(weights, bias)
      val dataPath = new Path(path, "data").toString
      sparkSession
        .createDataFrame(Seq(data))
        .repartition(1)
        .write
        .parquet(dataPath)
    }
  }

  private class LinearRegressionModelReader extends MLReader[LinearRegressionModel] {

    private val className = classOf[LinearRegressionModel].getName

    override def load(path: String): LinearRegressionModel = {
      // Загружаем метаданные и параметры
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      // Загружаем другие данные
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("weights", "bias").head()
      val weights = data.getAs[DenseVector](0)
      val bias = data.getAs[Double](1)

      // Создаём модель с загруженными параметрами
      val model = new LinearRegressionModel(metadata.uid, weights, bias)
      metadata.getAndSetParams(model)

      model
    }
  }
}



