import mpl.FeedForward
import utils.Matrix

@OptIn(ExperimentalStdlibApi::class)
suspend fun main() {
//    FeedForward().doFeedForward()
    Matrix.printMatrix(
        Matrix.transposeMatrix(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0)
            )
        )
    )
}