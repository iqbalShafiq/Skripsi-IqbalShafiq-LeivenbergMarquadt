import mpl.Backpropagation
import mpl.EpochRegulator
import mpl.EpochRegulator.startEpoch
import mpl.FeedForward
import testing.Testing
import utils.Matrix.printMatrix
import utils.Matrix.printWeight

fun main() {
    // starting mlp
    var currentMSE = 999.0
    var newMiu = EpochRegulator.MIU
    var inputWeightData: MutableList<MutableList<Double>> = mutableListOf()
    var lastInputWeightData: MutableList<MutableList<Double>> = mutableListOf()
    var hiddenWeightData: MutableList<MutableList<Double>> = mutableListOf()
    var lastHiddenWeightData: MutableList<MutableList<Double>> = mutableListOf()

    // Start Training
    startEpoch { maxEpoch, errorTarget ->
        for (currentEpoch in 1..maxEpoch) {
            if (currentMSE > errorTarget) {
                println("Starting Epoch-$currentEpoch\n")
                lastInputWeightData = inputWeightData
                lastHiddenWeightData = hiddenWeightData
                val feedForwardResult = FeedForward(false).startFeedForward(
                    newMiu,
                    inputWeightData,
                    hiddenWeightData
                )

                with(feedForwardResult) {
                    inputLayer.features.forEachIndexed { index, input ->
                        val backpropagation = Backpropagation(
                            input,
                            hiddenLayer[index],
                            activatedHiddenLayer[index],
                            outputLayer[index],
                            inputWeight,
                            hiddenWeight,
                            errorList,
                            newMiu
                        ).startBackpropagation()

                        if (index == inputLayer.features.size - 1) {
                            inputWeightData.clear()
                            hiddenWeightData.clear()
                            inputWeightData = backpropagation.inputHiddenWeight as MutableList<MutableList<Double>>
                            hiddenWeightData = backpropagation.hiddenOutputWeight as MutableList<MutableList<Double>>

                            println("New Vjk:")
                            printWeight(inputWeightData)

                            println("New Wjk:")
                            printWeight(hiddenWeightData)

                            currentMSE = feedForwardResult.mse
                            println("Latest MSE: " + feedForwardResult.mse)
                        }
                    }

                    newMiu = EpochRegulator.getNewLMParameter(
                        currentMSE, mse, miu
                    )
                }
            } else {
                println("Training berhenti pada epoch ke-${currentEpoch - 1}")
                break
            }
        }
    }

    // Start Testing
    val testingResult = Testing().startTesting(inputWeightData, hiddenWeightData)
    println(testingResult)
}