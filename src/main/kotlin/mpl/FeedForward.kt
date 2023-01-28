package mpl

import data.ErrorResult.getResultBias
import data.ErrorResult.getResultRecord
import data.FeedForwardResult
import data.InputData
import utils.ActivationFunction
import utils.ErrorValue
import utils.Matrix

class FeedForward(private val isTesting: Boolean) {

    fun startFeedForward(
        miu: Double,
        vjk: MutableList<MutableList<Double>> = mutableListOf(),
        wjk: MutableList<MutableList<Double>> = mutableListOf(),
        filePath: String = "src/data.csv"
    ): FeedForwardResult {
        val inputLayer = getInputData(filePath)

        val inputWeightData = if (vjk.isNotEmpty()) {
            val weight = mutableListOf<MutableList<Double>>()
            vjk.forEach { rows ->
                val record = mutableListOf<Double>()

                rows.forEachIndexed { index, value ->
                    if (index != vjk.first().size - 1) {
                        record.add(value)
                    }
                }

                weight.add(record)
            }

            weight
        } else {
            initWeightData(INPUT_LAYER_NEURON, HIDDEN_LAYER_NEURON)
        }

        val inputBiasData = if (vjk.isNotEmpty()) vjk.last() else {
            initBiasData(HIDDEN_LAYER_NEURON)
        }
        val hiddenWeightData = if (vjk.isNotEmpty()) {
            val weight = mutableListOf<MutableList<Double>>()
            wjk.forEach { rows ->
                val record = mutableListOf<Double>()

                rows.forEachIndexed { index, value ->
                    if (index != wjk.first().size - 1) {
                        record.add(value)
                    }
                }

                weight.add(record)
            }

            weight
        } else {
            initWeightData(HIDDEN_LAYER_NEURON, OUTPUT_NEURON)
        }

        val hiddenBiasData = if (wjk.isNotEmpty()) wjk.last() else {
            initBiasData(OUTPUT_NEURON)
        }

        val inputWeight = mutableListOf<List<Double>>()
        inputWeightData.forEachIndexed { index, rows ->
            rows.add(inputBiasData[index])
            inputWeight.add(rows)
        }

        val hiddenWeight = mutableListOf<List<Double>>()
        hiddenWeightData.forEachIndexed { index, rows ->
            rows.add(hiddenBiasData[index])
            hiddenWeight.add(rows)
        }

        val hiddenLayer = calculateNextLayer(
            inputLayer.features,
            inputWeightData,
            inputBiasData
        )

        val activatedHiddenLayer = calculateActivatedNet(hiddenLayer)

        val outputLayer = calculateNextLayer(
            hiddenLayer,
            hiddenWeightData,
            hiddenBiasData
        )

        val activatedOutputLayer = calculateActivatedNet(outputLayer)

        val errorList = getListOutputError(
            inputLayer.target,
            activatedOutputLayer.first()
        )

        val mse = ErrorValue.calculateMeanSquaredError(errorList)

        return FeedForwardResult(
            inputLayer, hiddenLayer,
            activatedHiddenLayer, outputLayer,
            inputWeight,
            hiddenWeight,
            activatedOutputLayer,
            errorList, mse, miu
        )
    }

    /**
     * Melakukan normalisasi dataset dengan membagi setiap pixel dengan 255
     * @return matrix m x n
     */
    fun getInputData(filePath: String): InputData {
        // read data from csv
        val data = Matrix.readCsvFile(filePath)

        val normalizedData = mutableListOf<List<Double>>()
        var targetData = mutableListOf<Double>()
        data.forEach { record ->
            // get last position
            val lastPosition = record.size - 1
            val newRecord = mutableListOf<Double>()

            record.forEachIndexed { index, value ->
                // exclude target
                if (index != lastPosition) {
                    newRecord.add(value / NORMALIZATION_DIVIDER)
                } else {
                    // target data
                    targetData.add(value)
                }
            }

            // add new record into normalized data
            normalizedData.add(newRecord)
        }

        return InputData(normalizedData, targetData)
    }

    /**
     * Inisialisasi beban secara random dari layer n-1 ke layer n
     * @param startLayer = layer ke n-1
     * @param targetLayer = layer ke n
     * @return matrix bias m x n
     */
    private fun initWeightData(startLayer: Int, targetLayer: Int): MutableList<MutableList<Double>> {
        val weights = mutableListOf<MutableList<Double>>()

        for (start in 1..targetLayer) {
            val record = mutableListOf<Double>()

            // create random number of weight from 0 to 1  for each branch
            for (target in 1..startLayer) {
                record.add(getResultRecord())
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
            biases.add(getResultBias())
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

                record.forEachIndexed { recordIndex, recordData ->
                    net += recordData * weights[weightRecordIndex][recordIndex]
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
                    ActivationFunction.calculateSigmoidFunction(net, isTesting)
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
        private const val INPUT_LAYER_NEURON = 13
        private const val HIDDEN_LAYER_NEURON = 1
        private const val OUTPUT_NEURON = 1
        private const val NORMALIZATION_DIVIDER = 255
    }
}