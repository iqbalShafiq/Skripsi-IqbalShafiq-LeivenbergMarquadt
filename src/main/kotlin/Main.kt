import mpl.FeedForward
import utils.Matrix

@OptIn(ExperimentalStdlibApi::class)
suspend fun main() {
//    FeedForward().doFeedForward()
    Matrix.printMatrix(
        Matrix.timesSquareMatrix(
            listOf(
                listOf(1.0, 2.0, 3.0),
                listOf(4.0, 5.0, 6.0),
                listOf(7.0, 8.0, 9.0)
            ),
            listOf(
                listOf(9.0, 8.0, 7.0),
                listOf(6.0, 5.0, 4.0),
                listOf(3.0, 2.0, 1.0)
            )
        )
    )
}