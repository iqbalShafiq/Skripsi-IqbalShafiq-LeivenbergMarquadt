package utils

import kotlin.math.exp

object ActivationFunction {

    /**
     * Menghitung fungsi aktivasi berdasarkan net yang
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateActivationFunction(net: Double): Double = 1 / (1 + exp(net))
}