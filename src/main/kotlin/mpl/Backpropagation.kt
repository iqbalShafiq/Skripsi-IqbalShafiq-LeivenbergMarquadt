package mpl

import data.WeightResult
import utils.ActivationFunction.calculateDerivativeSigmoidFunction
import utils.Matrix.calculateMatrixIdentity
import utils.Matrix.calculatePseudoInverse
import utils.Matrix.createHessianMatrix
import utils.Matrix.sumTwoMatrix
import utils.Matrix.timesConstWithMatrix
import utils.Matrix.timesMatrixWithColumnMatrix

class Backpropagation(
    private val inputLayer: List<Double>,
    private val hiddenNetLayer: List<Double>,
    private val hiddenLayer: List<Double>,
    private val inputWeight: List<List<Double>>,
    private val hiddenWeight: List<List<Double>>,
    private val targetData: List<List<Double>>,
    private val errorList: List<Double>,
    private val miu: Double
) {

    fun startBackpropagation() {
        // hidden ~ output
        val errorOutputLayer = calculateErrorOutputLayerNeuron(
            hiddenNetLayer,
            errorList
        )
        val correctionHiddenOutputWeight = calculateCorrectionWeight(
            hiddenLayer,
            errorOutputLayer
        )
        val correctionHiddenOutputBias = correctionHiddenOutputWeight.map { corrections ->
            corrections.last()
        }

        // input ~ hidden
        val errorNetHiddenLayer = calculateNetErrorHiddenLayerNeuron(
            errorOutputLayer,
            hiddenWeight
        )
        val errorHiddenLayer = calculateErrorHiddenLayerNeuron(
            errorNetHiddenLayer
        )
        val correctionInputHiddenWeight = calculateCorrectionWeight(
            inputLayer,
            errorHiddenLayer
        )
        val correctionInputHiddenBias = correctionInputHiddenWeight.map { corrections ->
            corrections.last()
        }

        // create jacobian
        val jacobianMatrix = getMatrixJacobian(
            correctionHiddenOutputWeight,
            correctionHiddenOutputBias,
            correctionInputHiddenWeight,
            correctionInputHiddenBias
        )

        // calculate Δw and Δv for update weight
        val correctionAllWeightAndBias = listOf<Double>()

        // set form of delta weight and bias
        val formedMatrixDeltaWeight = setDeltaWeightMatrixForm(correctionAllWeightAndBias)

        // get corrected weight and bias
        val correctedInputHiddenWeight = mutableListOf<List<Double>>()
        val correctedHiddenOutputWeight = mutableListOf<List<Double>>()

        formedMatrixDeltaWeight.forEachIndexed { index, rows ->
            if (index < inputWeight.size) {
                correctedInputHiddenWeight.add(rows)
            } else {
                correctedHiddenOutputWeight.add(rows)
            }
        }

        // get the update of weight and bias
        val updatedInputHiddenWeight = updateWeightData(
            inputWeight,
            correctedInputHiddenWeight
        )

        val updatedHiddenOutputWeight = updateWeightData(
            hiddenWeight,
            correctedHiddenOutputWeight
        )

        // return weight result
//        return WeightResult(
//            updatedInputHiddenWeight,
//            updatedHiddenOutputWeight
//        )
    }

    /**
     * Menghitung error pada setiap neuron di output layer
     * @param netList: sinyal input dari hidden layer dengan tipe data double
     * @param errorList each as error: target kelas dengan tipe data double
     * @return δ2 = error * f'(net) dengan δ (delta) bertipe data double
     */
    private fun calculateErrorOutputLayerNeuron(
        netList: List<Double>,
        errorList: List<Double>
    ): List<Double> {
        val deltaResult = mutableListOf<Double>()

        netList.forEachIndexed { index, net ->
            deltaResult.add(errorList[index] * calculateDerivativeSigmoidFunction(net))
        }

        return deltaResult
    }

    /**
     * Menghitung error pada setiap neuron di hidden layer
     * @param errorOutputLayer: matriks informasi error pada output layer
     * @param weightList: matriks beban dari hidden ke output layer
     * @return δ_in = δ_in * f'(δ_in) bertipe data double
     */
    private fun calculateNetErrorHiddenLayerNeuron(
        errorOutputLayer: List<Double>,
        weightList: List<List<Double>>
    ): List<Double> {
        val errorNetHiddenLayer = mutableListOf<Double>()

        weightList.forEachIndexed { index, rows ->
            var net = 0.0

            rows.forEach { weight ->
                net += errorOutputLayer[index] * weight
            }

            errorNetHiddenLayer.add(net)
        }

        return errorNetHiddenLayer
    }

    /**
     * Menghitung error pada setiap neuron di hidden layer
     * @param errorNetList: sinyal input dari error output layer
     * @return δ1 = δ_in * f'(δ_in) bertipe data double
     */
    private fun calculateErrorHiddenLayerNeuron(
        errorNetList: List<Double>
    ): List<Double> {
        val deltaResult = mutableListOf<Double>()

        errorNetList.forEach { net ->
            deltaResult.add(net * calculateDerivativeSigmoidFunction(net))
        }

        return deltaResult
    }

    /**
     * Menghitung koreksi beban dari setiap signal layer ke target layer
     * @param signalLayer z_j or x_j: matriks m x 1 dari layer sebelumnya bertipe data double
     * @param errorNeuron δ2_k or δ1_k: matriks n x 1 dari error pada layer
     * @return φ_jk = δ_k * z_j
     */
    private fun calculateCorrectionWeight(
        signalLayer: List<Double>,
        errorNeuron: List<Double>
    ): List<List<Double>> {
        val correctionWeight = mutableListOf<List<Double>>()

        errorNeuron.forEach { error ->
            val correction = mutableListOf<Double>()
            signalLayer.forEach { signal ->
                correction.add(error * signal)
            }

            correctionWeight.add(correction)
        }

        return correctionWeight
    }

    /**
     * Membentuk matriks jacobian
     * @param correctionHiddenOutputWeight as Δw_jk: koreksi bobot hidden ke output
     * @param correctionHiddenOutputBias as Δb2_k: koreksi bias hidden ke output
     * @param correctionInputHiddenWeight as Δv_jk: koreksi bobot input ke hidden
     * @param correctionInputHiddenBias as Δb1_j: koreksi bias input ke hidden
     * @return matrixJ = matriks jacobian m x 1
     */
    private fun getMatrixJacobian(
        correctionHiddenOutputWeight: List<List<Double>>,
        correctionHiddenOutputBias: List<Double>,
        correctionInputHiddenWeight: List<List<Double>>,
        correctionInputHiddenBias: List<Double>
    ): List<List<Double>> {
        val matrixJ = mutableListOf<List<Double>>()
        val row = mutableListOf<Double>()

        // add correction Δw_jk
        correctionHiddenOutputWeight.forEach { rows ->
            rows.forEach { weight -> row.add(weight) }
        }

        // add correction Δb2_k
        correctionHiddenOutputBias.forEach { bias -> row.add(bias) }

        // add correction Δv_jk
        correctionInputHiddenWeight.forEach { rows ->
            rows.forEach { weight -> row.add(weight) }
        }

        // add correction Δb1_k
        correctionInputHiddenBias.forEach { bias -> row.add(bias) }

        // add one row into matrixJ to create m x 1 matriks
        matrixJ.add(row)

        return matrixJ
    }

    /**
     * @param pseudoInverseMatrix = matriks pseudo inverse m x n dengan tipe data double
     * @param outputError as e = matriks error output jaringan m x 1 dengan tipe data double
     * @return Δw = pseudoInverse * e
     */
    private fun calculateDeltaWeight(
        pseudoInverseMatrix: List<List<Double>>,
        outputError: List<Double>
    ): List<Double> = timesMatrixWithColumnMatrix(pseudoInverseMatrix, outputError)

    /**
     * @param deltaWeight as Δw: matriks koreksi bobot & bias m x 1
     * @return Δw dalam bentuk matriks m x n
     */
    private fun setDeltaWeightMatrixForm(
        deltaWeight: List<Double>
    ): List<List<Double>> {
        val newFormDeltaWeight = mutableListOf<MutableList<Double>>()
        var index = 0

        // correction weight from input to hidden layer (v)
        inputWeight.forEach { rows ->
            val newRow = mutableListOf<Double>()

            rows.forEachIndexed { vIndex, _ ->
                // exclude for bias (b1)
                if (vIndex != rows.size - 1) {
                    newRow.add(deltaWeight[index])
                    index++
                }
            }

            newFormDeltaWeight.add(newRow)
        }

        // correction bias from input to hidden layer (b1)
        newFormDeltaWeight.forEach { rows ->
            rows.add(deltaWeight[index])
            index++
        }

        // correction weight from hidden to output layer (w)
        hiddenWeight.forEach { rows ->
            val newRow = mutableListOf<Double>()

            rows.forEachIndexed { wIndex, _ ->
                // exclude for bias (b2)
                if (wIndex != rows.size - 1) {
                    newRow.add(deltaWeight[index])
                    index++
                }
            }

            newFormDeltaWeight.add(newRow)
        }

        // correction bias from hidden to output layer (b2)
        newFormDeltaWeight.forEachIndexed { rowIndex, rows ->
            // get the first row of w in the matrix
            if (rowIndex >= inputWeight.size) {
                rows.add(deltaWeight[index])
                index++
            }
        }

        return newFormDeltaWeight
    }

    /**
     * Menghitung perbaikan bobot
     * @param lastWeight as w_lama = matriks bobot m x n dengan tipe data double
     * @param deltaWeight as Δw = matriks bobot m x n dengan tipe data double
     * @return w_baru = w_lama - Δw
     */
    private fun updateWeightData(
        lastWeight: List<List<Double>>,
        deltaWeight: List<List<Double>>
    ): List<List<Double>> {
        return sumTwoMatrix(
            lastWeight,
            timesConstWithMatrix(
                -1.0,
                deltaWeight
            )
        )
    }
}