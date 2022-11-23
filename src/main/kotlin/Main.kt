import mpl.FeedForward
import utils.Matrix

@OptIn(ExperimentalStdlibApi::class)
suspend fun main() {
//    FeedForward().doFeedForward()
    Matrix.printMatrix(
        Matrix.timesNonSquareMatrix(
            listOf(
                listOf(1.0, -7.0),
                listOf(5.0, 9.0)
            ),
            listOf(
                listOf(5.0, -3.0, 8.0),
                listOf(0.0, 2.0, -1.0)
            )
        )
    )
}