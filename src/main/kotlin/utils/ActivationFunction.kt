package utils

import kotlin.math.exp

object ActivationFunction {

    /**
     * Menghitung fungsi aktivasi sigmoid berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateSigmoidFunction(net: Double): Double = 1 / (1 + exp(net))

    /**
     * Menghitung turunan pertama fungsi aktivasi sigmoid berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateDerivativeSigmoidFunction(net: Double): Double =
        calculateSigmoidFunction(net) * (1 - calculateSigmoidFunction(net))
}