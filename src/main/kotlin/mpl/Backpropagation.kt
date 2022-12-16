package mpl

import utils.ActivationFunction.calculateDerivativeSigmoidFunction
import utils.Matrix.calculateMatrixIdentity
import utils.Matrix.calculatePseudoInverse
import utils.Matrix.createHessianMatrix
import utils.Matrix.sumTwoMatrix
import utils.Matrix.timesConstWithMatrix
import utils.Matrix.timesMatrixWithColumnMatrix

class Backpropagation {

    fun startBackpropagation(
        inputNetLayer: List<Double>,
        inputLayer: List<Double>,
        hiddenNetLayer: List<Double>,
        hiddenLayer: List<Double>,
        inputWeight: List<List<Double>>,
        hiddenWeight: List<List<Double>>,
        errorList: List<Double>,
        miu: Double
    ) {
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

        // calculate levenberg marquardt formula for update weight
        val levenbergMarquardtFormula = calculateDeltaWeight(
            calculatePseudoInverse(
                createHessianMatrix(jacobianMatrix),
                jacobianMatrix,
                calculateMatrixIdentity(jacobianMatrix.first().size),
                miu
            ),
            errorOutputLayer
        )
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
     * @param signalLayer as z_j or x_j: matriks m x 1 dari layer sebelumnya bertipe data double
     * @param errorNeuron as δ2_k or δ1_k: matriks n x 1 dari error pada layer
     * @return φ_jk = δ_k * z_j
     */
    fun calculateCorrectionWeight(
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
    fun getMatrixJacobian(
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
     * Menghitung perbaikan bobot
     * @param lastWeight as w_lama = matriks bobot m x n dengan tipe data double
     * @param deltaWeight as Δw = matriks bobot m x n dengan tipe data double
     * @return w_baru = w_lama - Δw
     */
    fun updateWeightData(
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