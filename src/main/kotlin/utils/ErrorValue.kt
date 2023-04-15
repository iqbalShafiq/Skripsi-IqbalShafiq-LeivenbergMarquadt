package utils

import kotlin.math.log

object ErrorValue {

    /**
     * Mendapatkan pola dari target
     * @param target target kelas (label) dengan tipe data double
     * @param currentIndex index output layer saat ini
     * @return one hot encoding dari target. Contoh: target = 3 -> 0 0 0 1 0 0 0
     */
    private fun getOneHotEncodingTarget(target: Double, currentIndex: Int): Double {
        return if (currentIndex == target.toInt()) 1.0
        else 0.0
    }

    /**
     * Menghitung error pada output layer berdasarkan value pada output dan target
     * @param target target kelas (label) dengan tipe data double
     * @param outputValue value dari output layer dengan tipe data double
     * @param currentIndex index output layer saat ini
     * @return mengembalikan error = target - outputValue dengan tipe data double
     */
    fun calculateOutputLayerError(target: Double, outputValue: Double, currentIndex: Int): Double {
        return getOneHotEncodingTarget(target, currentIndex) - outputValue
    }

    /**
     * Menghitung Mean Squared Error (MSE)
     * @param errorList list error dari output layer dengan tipe data list of double
     * @return Sum(power(error, 2)) / n dengan tipe data double
     */
    fun calculateMeanSquaredError(errorList: List<Double>): Double {
        var sumOfSquaredError = 0.0
        val n = errorList.size

        errorList.forEach { error ->
            sumOfSquaredError += (error * error)
        }

        return sumOfSquaredError / n
    }

    /**
     * Menghitung Categorical Cross Entropy Loss (CCE)
     * @param outputValue nilai aktivasi pada index sesuai target dari output layer
     * @return -log(outputValue) dengan tipe data double
     */
    fun calculateCategoricalCrossEntropyLoss(
        outputValue: Double
    ): Double = -log(outputValue, 10.0)

    /**
     * Menghitung error berdasarkan target dengan aktual value
     * @param targetValueList list target dari data
     * @param activatedOutputLayer index dari actual value yang didapatkan dari feedforward
     * @return total error dari feedforward
     */
    fun calculateErrorValue(
        targetValueList: List<Int>,
        activatedOutputLayer: List<List<Double>>
    ): Double {
        var totalError = 0.0

        targetValueList.forEachIndexed { index, target ->
            val indexOfActualValue = activatedOutputLayer[index].indexOf(
                activatedOutputLayer[index].maxOf { it }
            )

            totalError += if (target != indexOfActualValue) 1 else 0
        }

        return (totalError / targetValueList.size) * 100
    }
}