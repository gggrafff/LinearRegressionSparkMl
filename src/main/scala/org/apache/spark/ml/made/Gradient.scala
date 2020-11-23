package org.apache.spark.ml.made

import breeze.linalg.sum
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
 * Базовый класс для рассчитывателей градиентов. Сделан по образцу класса Gradient из spark.mllib.
 */
abstract class Gradient private[made] extends Serializable {
  /**
   * Рассчитывает градиент и значение функции потерь для одной строки данных.
   *
   * @param data Признаки одного объекта данных
   * @param label Верное значение прогноза для одной строки данных
   * @param weights Веса/коэффициенты
   *
   * @return (gradient: Vector, loss: Double) Градиент и значение функции потерь
   */
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double)
}

/**
 * Рассчитывает градиенты и функцию потерь согласно https://media.geeksforgeeks.org/wp-content/uploads/Cost-Function.jpg
 */
class LeastSquaresGradient private[made] extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val diff = sum(data.asBreeze *:* weights.asBreeze) - label
    val loss = diff * diff / 2.0
    val gradient = data.copy.asBreeze * diff
    (Vectors.fromBreeze(gradient), loss)
  }
}
