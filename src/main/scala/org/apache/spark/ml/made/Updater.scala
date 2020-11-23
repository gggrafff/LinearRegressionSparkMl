package org.apache.spark.ml.made

import breeze.linalg.{Vector => BV, axpy => brzAxpy}
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
 * Базовый класс для классов, обновляющих веса. Сделан по образцу класса Updater из spark.mllib.
 */
abstract class Updater private[made] extends Serializable {
  /**
   * Рассчитывает новый вектор весов.
   * @param weightsOld Старый вектор весов.
   * @param gradient Градиент.
   * @param stepSize Learning rate.
   * @param iter Номер итерации обучения (для изменения lr в ходе обучения).
   * @return Новый вектор весов.
   */
  def compute(
               weightsOld: Vector,
               gradient: Vector,
               stepSize: Double,
               iter: Int): Vector
}

/**
 * Обновляет веса модели. Не изменяет lr по ходу обучения.
 */
class SimpleUpdater private[made] extends Updater {
  override def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       iter: Int): Vector = {
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    brzAxpy(-stepSize, gradient.asBreeze, brzWeights)

    Vectors.fromBreeze(brzWeights)
  }
}

/**
 * Обновляет веса модели. Уменьшает lr по ходу обучения.
 */
class ReducingStepsUpdater private[made] extends Updater {
  override def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       iter: Int): Vector = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.asBreeze, brzWeights)

    Vectors.fromBreeze(brzWeights)
  }
}