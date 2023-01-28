package testing

import data.TestingResult
import mpl.EpochRegulator
import mpl.FeedForward

class Testing {
    fun startTesting(
        vjk: MutableList<MutableList<Double>>,
        wjk: MutableList<MutableList<Double>>,
    ): TestingResult {
        val feedForward = FeedForward(true)
        val target = feedForward.getInputData("src/data.csv").target
        val result = feedForward.startFeedForward(
            EpochRegulator.MIU, vjk, wjk, "src/data.csv"
        )

        val accuracy = getAccuracy(target, result.activatedOutputLayer)

        return TestingResult(accuracy)
    }

    private fun getAccuracy(target: List<Double>, outputLayer: List<List<Double>>): Double {
        var totalTrue = 0

        target.forEachIndexed { index, value ->
            val output = outputLayer[index].first()

            if (value == output) totalTrue++
        }

        return (totalTrue / target.size.toDouble()) * 100
    }
}