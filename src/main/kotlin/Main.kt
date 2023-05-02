import mpl.FeedForward
import mpl.NewBackpropagation

fun main() {
    // init input ~ hidden weight and bias
    val inputWeight = FeedForward().initWeightData(
        FeedForward.INPUT_LAYER_NEURON,
        FeedForward.HIDDEN_LAYER_NEURON
    )
    val inputBias = FeedForward().initBiasData(
        FeedForward.HIDDEN_LAYER_NEURON
    )

    // init hidden ~ output weight and bias
    val hiddenWeight = FeedForward().initWeightData(
        FeedForward.HIDDEN_LAYER_NEURON,
        FeedForward.OUTPUT_NEURON
    )
    val hiddenBias = FeedForward().initBiasData(
        FeedForward.OUTPUT_NEURON
    )

    // get input layer
    val inputLayer = FeedForward().getInputData()

    // starting epoch
    val feedForwardResult = FeedForward().startFeedForward(
        inputLayer, inputWeight, inputBias, hiddenWeight, hiddenBias
    )

    var backproInputWeight: MutableList<MutableList<Double>>
    inputWeight.forEachIndexed { index, row ->
        row.add(inputBias[index])
    }
    backproInputWeight = inputWeight

    var backproHiddenWeight: MutableList<MutableList<Double>>
    hiddenWeight.forEachIndexed { index, row ->
        row.add(hiddenBias[index])
    }
    backproHiddenWeight = hiddenWeight

    var accuracy = feedForwardResult.accuracyValue
    var miu = 0.9
    var mse = 9.99
    val beta = 1.005
    var epochCounter = 0
    while (epochCounter < 10) {
        epochCounter++
        val weightResult = NewBackpropagation(
            feedForwardResult.targetList,
            feedForwardResult.inputLayer,
            feedForwardResult.signalHiddenLayer,
            feedForwardResult.hiddenLayer,
            feedForwardResult.signalOutputLayer,
            feedForwardResult.outputLayer,
            backproInputWeight,
            backproHiddenWeight,
            feedForwardResult.errorList,
            miu
        ).startBackpropagation()

        with(weightResult) {
            val newFeedForwardResult = FeedForward().startFeedForward(
                inputLayer,
                inputHiddenWeight,
                inputHiddenBias,
                hiddenOutputWeight,
                hiddenOutputBias
            )
            println("Epoch: $epochCounter")
            println("Miu: $miu")

            inputHiddenWeight.forEachIndexed { index, row ->
                row.add(inputHiddenBias[index])
            }
            backproInputWeight.clear()
            backproInputWeight = inputHiddenWeight

            hiddenOutputWeight.forEachIndexed { index, row ->
                row.add(hiddenOutputBias[index])
            }
            backproHiddenWeight.clear()
            backproHiddenWeight = hiddenOutputWeight

            with(feedForwardResult) {
                signalHiddenLayer = newFeedForwardResult.signalHiddenLayer
                hiddenLayer = newFeedForwardResult.hiddenLayer
                signalOutputLayer = newFeedForwardResult.signalOutputLayer
                outputLayer = newFeedForwardResult.outputLayer
                errorList = newFeedForwardResult.errorList
            }

            mse = newFeedForwardResult.mse
            if (mse < feedForwardResult.mse) {
                miu /= beta
            } else {
                miu *= beta
            }

            accuracy = newFeedForwardResult.accuracyValue
        }
    }
}