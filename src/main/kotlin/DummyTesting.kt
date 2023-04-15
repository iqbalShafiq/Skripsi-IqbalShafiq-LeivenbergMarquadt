import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import utils.ActivationFunction
import kotlin.math.exp
import kotlin.random.Random

fun main() {
    val record = mutableListOf<Double>()
    for (target in 0 until 100) {
        record.add(Random.nextDouble(0.0, 0.1))
    }

    println(record)
}