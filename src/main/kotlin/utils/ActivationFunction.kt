package utils

import kotlin.math.exp

object ActivationFunction {

    /**
     * Menghitung fungsi aktivasi sigmoid berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateSigmoidFunction(
        net: Double,
        isTesting: Boolean
    ): Double {
        val sigmoid = 1 / (1 + exp(-1 * net))

        return if (isTesting) {
            if (sigmoid < 0.5) 0.0 else 1.0
        } else sigmoid
    }

    /**
     * Menghitung turunan pertama fungsi aktivasi sigmoid berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateDerivativeSigmoidFunction(net: Double): Double =
        calculateSigmoidFunction(net, false) * (1 - calculateSigmoidFunction(net, false))
}