package org.apache.spark.ml.made

import breeze.linalg.norm
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.sql.expressions.Aggregator

import scala.collection.mutable.ArrayBuffer

/**
 * Характеристика для оптимизаторов. Сделана по образцу трейта Optimizer из spark.mllib.
 */
trait Optimizer extends Serializable {
  def optimize(dataset: Dataset[_], initialWeights: Vector): Vector
}

/**
 * Полный градиентный спуск.
 * @param gradient Объект для рассчёта градиентов.
 * @param updater Объект для обновления весов.
 * @param inputCol Имя входной колонки с признаками объектов.
 * @param outputCol Имя выходной колонки со значениями прогнозов.
 */
class GradientDescent private[made] (private var gradient: Gradient,
                                     private var updater: Updater,
                                     private var inputCol: String,
                                     private var outputCol: String)
  extends Optimizer with Logging{

  private var stepSize: Double = 0.001
  private var numIterations: Int = 100
  private var convergenceTol: Double = 0.001

  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
   * Установить допуск сходимости. Значение по умолчанию 0.001
   * convergenceTol - это параметр, на основании которого решается, когда завершить оптимизацию.
   * Окончание оптимизации определяется на основе приведенной ниже логики.
   *
   * - Если норма нового вектора решения больше 1, то разница векторов решения
   * сравнивается с относительным допуском, что означает нормализацию по норме
   * нового вектора решения.
   * - Если норма нового вектора решения меньше или равна 1, то разница векторов
   * решения сравнивается с абсолютным допуском, который не нормализуется.
   *
   * Должно быть от 0.0 до 1.0 включительно.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  def setInputCol(inputCol: String): this.type = {
    this.inputCol = inputCol
    this
  }

  def setOutputCol(outputCol: String): this.type = {
    this.outputCol= outputCol
    this
  }

  /**
   * Запустить градиентный спуск на тренировочных данных.
   * @param dataset Тренировочный датасет.
   * @param initialWeights Начальный вектор весов.
   * @return Решение задачи оптимизации.
   */
  def optimize(dataset: Dataset[_], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescent.runGD(
      dataset,
      gradient,
      updater,
      stepSize,
      numIterations,
      initialWeights,
      convergenceTol,
      inputCol,
      outputCol)
    weights
  }
}

/**
 *
 */
object GradientDescent extends Logging {
  /**
   * Запускает градиентный спуск распределённо.
   *
   * @param dataset Тренировочный датасет (label, [feature values]).
   * @param gradient Объект, рассчитывающий градиенты и значения функции потерь.
   * @param updater Объект, обновляющий веса.
   * @param stepSize Learning rate
   * @param numIterations Максимальное количество итераций обучения.
   * @param convergenceTol Допуск сходимости. По умолчанию 0.001. Должен быть от 0.0 до 1.0 включительно.
   * @return Кортеж с оптимизированными весами модели и значения функции потерь по ходу обучения.
   */
  def runGD(dataset: Dataset[_],
            gradient: Gradient,
            updater: Updater,
            stepSize: Double,
            numIterations: Int,
            initialWeights: Vector,
            convergenceTol: Double,
            inputCol: String,
            outputCol: String): (Vector, Array[Double]) = {
    val iterationsNotChanged = 5

    val LossHistory = new ArrayBuffer[Double](numIterations)
    var bestLoss = Double.MaxValue
    var badItersCount = 0
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = dataset.count()

    // Если данных нет, возвращаем начальные веса
    if (numExamples == 0) {
      logWarning("GradientDescent.runGD returning initial weights, no data found")
      return (initialWeights, LossHistory.toArray)
    }

    var weights = Vectors.dense(initialWeights.toArray)
    var bestWeights = weights
    val weights_count = weights.size
    val rows_count = dataset.count()

    var converged = false  // индикатор того, что решение сошлось
    var i = 1
    while (!converged && i <= numIterations) {

      // Считаем суммарные градиент и потери
      val customSummer =  new Aggregator[Row, (Vector, Double), (Vector, Double)] {
        def zero: (Vector, Double) = (Vectors.zeros(weights_count), 0.0)
        def reduce(acc: (Vector, Double), x: Row): (Vector, Double) = {
          val (grad, loss) = gradient.compute(x.getAs[Vector](inputCol), x.getAs[Double](outputCol), weights)
          (Vectors.fromBreeze(acc._1.asBreeze + grad.asBreeze / rows_count.asInstanceOf[Double]), acc._2 + loss / rows_count.asInstanceOf[Double])
        }
        def merge(acc1: (Vector, Double), acc2: (Vector, Double)): (Vector, Double) = (Vectors.fromBreeze(acc1._1.asBreeze + acc2._1.asBreeze), acc1._2 + acc2._2)
        def finish(r: (Vector, Double)): (Vector, Double) = r
        override def bufferEncoder: Encoder[(Vector, Double)] = ExpressionEncoder()
        override def outputEncoder: Encoder[(Vector, Double)] = ExpressionEncoder()
      }.toColumn

      val row = dataset.select(customSummer.as[(Vector, Double)](ExpressionEncoder()))

      val loss = row.first()._2
      LossHistory += loss
      weights = updater.compute(weights, row.first()._1, stepSize, i)

      if (loss < bestLoss) {
        bestLoss = row.first()._2
        bestWeights = weights
        badItersCount = 0
      } else {
        badItersCount += 1
      }

      previousWeights = currentWeights
      currentWeights = Some(weights)
      if (previousWeights.isDefined && currentWeights.isDefined) {
        if (convergenceTol == 0.0){
          converged = badItersCount > iterationsNotChanged
        } else {
          converged = isConverged(previousWeights.get, currentWeights.get, convergenceTol)
        }
      }
      i += 1
    }

    logInfo("GradientDescent.runGD finished. Last 10 losses %s".format(LossHistory.takeRight(10).mkString(", ")))

    (weights, LossHistory.toArray)
  }

  /**
   * Определяем, сошлось ли решение.
   * @param previousWeights Предыдущий вектор весов
   * @param currentWeights Текущий вектор весов
   * @param convergenceTol Допуск сходимости
   * @return True, если решение найдено, иначе - false.
   */
   def isConverged(previousWeights: Vector,
                  currentWeights: Vector,
                  convergenceTol: Double): Boolean = {
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }

}