package mpl

import utils.ActivationFunction.calculateDerivativeSigmoidFunction
import utils.ActivationFunction.calculateSigmoidFunction
import utils.Matrix.sumTwoMatrix
import utils.Matrix.timesConstWithMatrix
import utils.Matrix.timesNonSquareMatrix

class Backpropagation {

    /**
     * Menghitung error pada setiap neuron di output layer
     * @param netList: sinyal input dari hidden layer dengan tipe data double
     * @param targetList: target kelas dengan tipe data double
     * @return δ2 = (targetList - f(net) * f'(net) dengan δ (delta) bertipe data double
     */
    fun calculateErrorOutputLayerNeuron(
        netList: List<Double>,
        targetList: List<Double>
    ): List<Double> {
        val deltaResult = mutableListOf<Double>()

        netList.forEachIndexed { index, net ->
            val difference = targetList[index] - calculateSigmoidFunction(net)
            deltaResult.add(difference * calculateDerivativeSigmoidFunction(net))
        }

        return deltaResult
    }

    /**
     * Menghitung error pada setiap neuron di hidden layer
     * @param netList: sinyal input dari input layer dengan tipe data double
     * @param targetList: target kelas dengan tipe data double
     * @return δ1 = δ_in * f'(δ_in) bertipe data double
     */
    fun calculateErrorHiddenLayerNeuron(
        netList: List<Double>,
        targetList: List<Double>
    ): List<Double> {
        val deltaResult = mutableListOf<Double>()

        netList.forEachIndexed { index, net ->
            val difference = targetList[index] - calculateSigmoidFunction(net)
            deltaResult.add(difference * calculateDerivativeSigmoidFunction(net))
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
     * @param pseudoInverseMatrix = matriks pseudo inverse m x n dengan tipe data double
     * @param outputError as e = matriks error output jaringan m x 1 dengan tipe data double
     * @return Δw = pseudoInverse * e
     */
    fun calculateDeltaWeight(
        pseudoInverseMatrix: List<List<Double>>,
        outputError: List<List<Double>>
    ): List<List<Double>> = timesNonSquareMatrix(pseudoInverseMatrix, outputError)

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