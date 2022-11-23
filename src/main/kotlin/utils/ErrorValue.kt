package utils

object ErrorValue {

    /**
     * Menghitung error pada output layer berdasarkan value pada output dan target
     * @param target = target kelas (label) dengan tipe data double
     * @param outputValue = value dari output layer dengan tipe data double
     * @return mengembalikan error = target - outputValue dengan tipe data double
     */
    fun calculateOutputLayerError(target: Double, outputValue: Double): Double {
        return target - outputValue
    }

    /**
     * Menghitung Mean Squared Error (MSE)
     * @param errorList = list error dari output layer dengan tipe data list of double
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
}