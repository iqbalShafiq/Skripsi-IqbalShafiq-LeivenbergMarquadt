package mpl

import data.InputData
import utils.ActivationFunction
import utils.ErrorValue
import utils.Matrix
import kotlin.random.Random

class FeedForward {

    suspend fun startFeedForward() {
        val inputLayer = getInputData()

        val hiddenLayer = calculateNextLayer(
            inputLayer.features,
            initWeightData(INPUT_LAYER_NEURON, HIDDEN_LAYER_NEURON),
            initBiasData(HIDDEN_LAYER_NEURON)
        )

        val activatedHiddenLayer = calculateActivatedNet(hiddenLayer)

        val outputLayer = calculateNextLayer(
            hiddenLayer,
            initWeightData(HIDDEN_LAYER_NEURON, OUTPUT_NEURON),
            initBiasData(OUTPUT_NEURON)
        )

        val activatedOutputLayer = calculateActivatedNet(outputLayer)

        val errorList = outputLayer.mapIndexed { index, output ->
            getListOutputError(
                inputLayer.target[index],
                output
            )
        }

        val flattenErrorList = flatFormErrorList(errorList)

        println("Input Layer:\n$inputLayer\n")
        println("Hidden Layer:\n$hiddenLayer\n")
        println("Output Layer:\n$outputLayer")
    }

    /**
     * Melakukan normalisasi dataset dengan membagi setiap pixel dengan 255
     * @return matrix m x n
     */
    private suspend fun getInputData(): InputData {
        // read data from csv
        val data = Matrix.readCsvFile()

        val normalizedData = mutableListOf<List<Double>>()
        val targetData = mutableListOf<List<Double>>()
        data.forEach { record ->
            // get last position
            val lastPosition = record.size - 1
            val newRecord = mutableListOf<Double>()

            // divide every pixel with 255
            record.forEachIndexed { index, pixelValue ->
                // exclude label
                if (index != lastPosition) {
                    newRecord.add(pixelValue / NORMALIZATION_DIVIDER)
                } else {
                    // encode target
                    targetData.add(encodeClass(pixelValue))
                }
            }

            // add new record into normalized data
            normalizedData.add(newRecord)
        }

        return InputData(normalizedData, targetData)
    }

    private fun encodeClass(target: Double): MutableList<Double> {
        val targetValue = mutableListOf<Double>()

        for (i in 0 until OUTPUT_NEURON) {
            if (i == target.toInt()) targetValue.add(1.0)
            else targetValue.add(0.0)
        }

        return targetValue
    }

    /**
     * Inisialisasi beban secara random dari layer n-1 ke layer n
     * @param startLayer = layer ke n-1
     * @param targetLayer = layer ke n
     * @return matrix bias m x n
     */
    private fun initWeightData(startLayer: Int, targetLayer: Int): List<List<Double>> {
        val weights = mutableListOf<List<Double>>()

        for (start in 1..targetLayer) {
            val record = mutableListOf<Double>()

            // create random number of weight from 0 to 1  for each branch
            for (target in 1..startLayer) {
                record.add(Random.nextDouble(0.0, 1.0))
            }

            weights.add(record)
        }

        return weights
    }

    /**
     * Inisialisasi bias secara random pada suatu layer
     * @param neuron = banyak neuron
     * @return matrix bias m x n
     */
    private fun initBiasData(neuron: Int): List<Double> {
        val biases = mutableListOf<Double>()

        // create random number of bias from 0 to 1 for each branch
        for (start in 1..neuron) {
            biases.add(Random.nextDouble(0.0, 1.0))
        }

        return biases
    }

    /**
     * Menghitung proses feedforward dari layer n-1 ke layer n
     * dengan rumus: net = Sum(weight * input) + bias
     * @param startLayer = nilai input dari layer n-1
     * @param weights = bobot dari layer n-1 ke layer n
     * @param biases = bias dari layer n-1 ke layer n
     * @return hasil dari f(net) = 1 / (1 + exp(net))
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

    private fun calculateActivatedNet(
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

    private fun getListOutputError(
        targetList: List<Double>,
        outputLayer: List<Double>
    ): List<Double> {
        val errorList = mutableListOf<Double>()

        outputLayer.forEachIndexed { index, output ->
            errorList.add(
                ErrorValue.calculateOutputLayerError(
                    targetList[index],
                    output
                )
            )
        }

        return errorList
    }

    private fun flatFormErrorList(errorList: List<List<Double>>): List<Double> {
        val flattenErrors = mutableListOf<Double>()

        errorList.forEach { rows ->
            rows.forEach { error ->
                flattenErrors.add(error)
            }
        }

        return flattenErrors
    }

    companion object {
        private const val INPUT_LAYER_NEURON = 2
        private const val HIDDEN_LAYER_NEURON = 4
        private const val OUTPUT_NEURON = 7
        private const val MAX_EPOCH = 5
        private const val ERROR_TARGET = 0.5
        private const val NORMALIZATION_DIVIDER = 255
    }
}