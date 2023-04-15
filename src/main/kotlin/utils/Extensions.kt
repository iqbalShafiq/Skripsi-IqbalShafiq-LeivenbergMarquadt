package utils

import kotlin.math.abs

object Extensions {
    fun Double.equalsDelta(other: Double) = abs(this - other) < 0.000001
    fun Collection<Double>.toDoubleArray(): DoubleArray {
        val result = DoubleArray(size)
        var index = 0
        for (element in this)
            result[index++] = element
        return result
    }
}