package mpl

import data.FeedForwardResult
import data.InputData
import utils.ActivationFunction
import utils.ErrorValue
import utils.Matrix
import kotlin.random.Random

class FeedForward {

    fun startFeedForward(
        inputLayer: InputData,
        inputHiddenWeights: List<List<Double>>,
        inputHiddenBiases: List<Double>,
        hiddenOutputWeights: List<List<Double>>,
        hiddenOutputBiases: List<Double>,
    ): FeedForwardResult {

        val hiddenLayer = calculateNextLayer(
            inputLayer.features,
            inputHiddenWeights,
            inputHiddenBiases
        )

        val activatedHiddenLayer = calculateActivatedHiddenNet(hiddenLayer)

        val outputLayer = calculateNextLayer(
            activatedHiddenLayer,
            hiddenOutputWeights,
            hiddenOutputBiases
        )

        val activatedOutputLayer = calculateActivatedOutputNet(outputLayer)
        println("Features: ${inputLayer.features[0].size}")
        println("Target: ${inputLayer.target.size}")
        val errorList = activatedOutputLayer.mapIndexed { index, output ->
            getListOutputError(
                inputLayer.target[index],
                output
            )
        }

        val flattenErrorList = flatFormErrorList(errorList)

        println("flattenErrorList: ${flattenErrorList.size}")
        println("Distinct flattenErrorList: ${flattenErrorList.distinct().size}")

        val errorValue = ErrorValue.calculateErrorValue(
            inputLayer.target.map { it.toInt() },
            activatedOutputLayer
        )
        val accuracyValue = 100 - errorValue
        val mse = ErrorValue.calculateMeanSquaredError(flattenErrorList)

        println("Input Layer:\n$inputLayer\n")
        println("Input ~ Hidden Weight:\n$inputHiddenWeights\n")
        println("Input ~ Hidden Bias:\n$inputHiddenBiases\n")
        println("Hidden Layer:\n$hiddenLayer\n")
        println("Activated Hidden Layer:\n$activatedHiddenLayer\n")
        println("Hidden ~ Output Weight:\n$hiddenOutputWeights\n")
        println("Hidden ~ Output Bias:\n$hiddenOutputBiases\n")
        println("Output Layer:\n$outputLayer\n")
        println("Activated Output Layer:\n$activatedOutputLayer\n")
        println("Error Values:\n$errorList\n")
        println("Flatten Error Values:\n$flattenErrorList\n")
        println("Error Accuracy:\n$errorValue\n")
        println("Accuracy:\n$accuracyValue\n")
        println("MSE:\n$mse\n")

        return FeedForwardResult(
            inputLayer.target,
            inputLayer.features,
            hiddenLayer,
            activatedHiddenLayer,
            outputLayer,
            activatedOutputLayer,
            flattenErrorList,
            errorValue,
            accuracyValue,
            mse
        )
    }

    /**
     * Melakukan normalisasi dataset dengan membagi setiap pixel dengan 255
     * @return matrix m x n
     */
    fun getInputData(): InputData {
        // read data from csv
        val data = Matrix.readCsvFile().filterIndexed { index, _ ->
            index < 3500
        }

        val normalizedData = mutableListOf<List<Double>>()
        val targetData = mutableListOf<Double>()

        data.forEach { record ->
            val lastPosition = record.size - 1
            val newRecord = mutableListOf<Double>()

            // divide every pixel with 256
            record.forEachIndexed { index, pixelValue ->
                if (index == lastPosition) {
                    targetData.add(pixelValue)
                } else {
                    newRecord.add(pixelValue / NORMALIZATION_DIVIDER)
                }
            }

            // add new record into normalized data
            normalizedData.add(newRecord)
        }

        return InputData(normalizedData, targetData)
    }

    /**
     * Inisialisasi beban secara random dari layer n-1 ke layer n
     * @param startLayer layer ke n-1
     * @param targetLayer layer ke n
     * @return matrix bias m x n
     */
    fun initWeightData(startLayer: Int, targetLayer: Int): MutableList<MutableList<Double>> {
        val weights = mutableListOf<MutableList<Double>>()

        for (start in 0 until targetLayer) {
            val record = mutableListOf<Double>()

            // create random number of weight from 0 to 1  for each branch
            for (target in 0 until startLayer) {
                record.add(Random.nextDouble(0.0, 1.0))
            }

            weights.add(record)
        }

        return weights
    }

    /**
     * Inisialisasi bias secara random pada suatu layer
     * @param neuron banyak neuron
     * @return matrix bias m x n
     */
    fun initBiasData(neuron: Int): MutableList<Double> {
        val biases = mutableListOf<Double>()

        // create random number of bias from 0 to 1 for each branch
        for (start in 0 until neuron) {
            biases.add(Random.nextDouble(0.0, 1.0))
        }

        return biases
    }

    /**
     * Menghitung proses feedforward dari layer n-1 ke layer n
     * dengan rumus: net = Sum(weight * input) + bias
     * @param startLayer nilai input dari layer n-1
     * @param weights bobot dari layer n-1 ke layer n
     * @param biases bias dari layer n-1 ke layer n
     * @return hasil dari net = bias + sum(startLayerValue * weight)
     */
    private fun calculateNextLayer(
        startLayer: List<List<Double>>,
        weights: List<List<Double>>,
        biases: List<Double>
    ): List<List<Double>> {
        val nextLayer = mutableListOf<List<Double>>()

        startLayer.forEach { record ->
            val newRecord = mutableListOf<Double>()

            weights.forEachIndexed { weightRecordIndex, weightRecord ->
                var net = 0.0

                record.forEachIndexed { recordDataIndex, recordData ->
                    net += recordData * weightRecord[recordDataIndex]
                }

                net += biases[weightRecordIndex]
                newRecord.add(net)
            }

            nextLayer.add(newRecord)
        }

        return nextLayer
    }

    private fun calculateActivatedHiddenNet(
        netList: List<List<Double>>
    ): List<List<Double>> {
        val activatedNet = mutableListOf<List<Double>>()

        netList.forEach { rows ->
            val row = mutableListOf<Double>()

            rows.forEach { net ->
                row.add(
                    ActivationFunction.calculateSigmoidFunction(net)
                )
            }

            activatedNet.add(row)
        }

        return activatedNet
    }

    private fun calculateActivatedOutputNet(
        netList: List<List<Double>>
    ): List<List<Double>> {
        val activatedNet = mutableListOf<List<Double>>()

        netList.forEach { rows ->
            val row = mutableListOf<Double>()

            rows.forEach { net ->
                row.add(
                    ActivationFunction.calculateSoftmaxFunction(net, rows)
                )
            }

            activatedNet.add(row)
        }

        return activatedNet
    }

    private fun getListOutputError(
        target: Double,
        outputLayer: List<Double>
    ): List<Double> {
        val errorList = mutableListOf<Double>()

        outputLayer.forEachIndexed { index, outputValue ->
            errorList.add(
                ErrorValue.calculateOutputLayerError(target, outputValue, index)
            )
        }

        return errorList
    }

    private fun flatFormErrorList(errorList: List<List<Double>>): List<Double> {
        val flattenErrors = mutableListOf<Double>()

        errorList.forEach { rows ->
            rows.forEach { rowValue ->
                flattenErrors.add(rowValue)
            }
        }

        return flattenErrors
    }

    companion object {
        const val INPUT_LAYER_NEURON = 4
        const val HIDDEN_LAYER_NEURON = 5
        const val OUTPUT_NEURON = 4
        private const val MAX_EPOCH = 5
        private const val ERROR_TARGET = 0.5
        private const val NORMALIZATION_DIVIDER = 256
    }
}