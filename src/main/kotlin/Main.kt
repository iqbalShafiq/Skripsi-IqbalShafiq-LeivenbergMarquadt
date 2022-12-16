import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import mpl.EpochRegulator.startEpoch
import mpl.FeedForward

suspend fun main() {
    // starting epoch
    startEpoch {
        CoroutineScope(Dispatchers.IO).launch {
            FeedForward().startFeedForward()
        }
    }
}