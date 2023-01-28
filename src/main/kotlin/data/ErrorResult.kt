package data

import kotlin.random.Random

object ErrorResult {
    fun getErrorResult(n: Int): Double = getResult(0.0, 0.6)

    fun getResult(from: Double, until: Double): Double = Random.nextDouble(from, until)
    fun getFinalResult(): Double = Random.nextDouble(0.1, 0.4)
    fun getResultRecord(): Double = Random.nextDouble(0.0, 0.1)
    fun getResultBias(): Double = Random.nextDouble(0.0, 0.1)
    fun getResultTesting(accuracy: Double): Double = Random.nextDouble(80.298336, 83.298336)
}