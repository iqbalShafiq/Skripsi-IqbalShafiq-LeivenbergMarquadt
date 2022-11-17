import mpl.FeedForward

@OptIn(ExperimentalStdlibApi::class)
suspend fun main() {
    FeedForward().doFeedForward()
}