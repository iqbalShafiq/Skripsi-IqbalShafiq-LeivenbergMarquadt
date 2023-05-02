package utils

import kotlin.math.exp
import kotlin.math.max

object ActivationFunction {

    /**
     * Menghitung fungsi aktivasi sigmoid berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateSigmoidFunction(net: Double): Double = 1 / (1 + exp(-net))

    /**
     * Menghitung turunan pertama fungsi aktivasi sigmoid berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */

    fun calculateDerivativeSigmoidFunction(net: Double): Double =
        calculateSigmoidFunction(net) * (1 - calculateSigmoidFunction(net))

    /**
     * Menghitung fungsi aktivasi RELu berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateRELuFunction(net: Double): Double = max(0.0, net)

    /**
     * Menghitung turunan dari fungsi aktivasi RELu berdasarkan net
     * @param net = input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateDerivativeRELuFunction(net: Double): Double = if (net >= 0.0) 1.0 else 0.0

    /**
     * Menghitung fungsi aktivasi softmax berdasarkan net
     * @param net = input dengan tipe data double
     * @param netLIst = list semua input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateSoftmaxFunction(net: Double, netList: List<Double>): Double {
        return exp(net) / netList.sumOf { exp(it) }
    }

    /**
     * Menghitung fungsi aktivasi softmax berdasarkan net
     * @param currentNet = input dengan tipe data double
     * @param netLIst = list semua input dengan tipe data double
     * @return mengembalikan nilai dengan tipe data double
     */
    fun calculateDerivativeSoftmaxFunction(
        currentNet: Double,
        selectedNet: Double,
        netList: List<Double>,
        isSameIndex: Boolean = true
    ): Double = if (isSameIndex) {
        calculateSoftmaxFunction(
            currentNet,
            netList
        ) * (1 - calculateSoftmaxFunction(selectedNet, netList))
    } else {
        -calculateSoftmaxFunction(
            currentNet,
            netList
        ) * calculateSoftmaxFunction(
            selectedNet,
            netList
        )
    }

}