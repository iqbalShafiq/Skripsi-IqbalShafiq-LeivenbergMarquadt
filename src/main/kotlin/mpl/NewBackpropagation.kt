package mpl

import data.WeightResult
import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import utils.ActivationFunction.calculateDerivativeSigmoidFunction
import utils.Matrix
import utils.Matrix.calculatePseudoInverse
import utils.Matrix.timesMatrixWithColumnMatrix

class NewBackpropagation(
    private val targetList: List<Double>, // t
    private val inputLayer: List<List<Double>>, // x
    private val signalHiddenLayer: List<List<Double>>, // z_in
    private val hiddenLayer: List<List<Double>>, // z
    private val signalOutputLayer: List<List<Double>>, // y_in
    private val outputLayer: List<List<Double>>, // y
    private val inputWeight: List<List<Double>>, // v
    private val hiddenWeight: List<List<Double>>, // w
    private val errorList: List<Double>, // e
    private val miu: Double
) {

    fun startBackpropagation(): WeightResult {
        // output ~ hidden
        val weightCorrectionHiddenOutput = mutableListOf<List<List<Double>>>()
        outputLayer.forEachIndexed { dataIndex, outputValuePerData ->
            val weightCorrectionHiddenOutputPerData = mutableListOf<List<Double>>()

            calculateWeightCorrectionOutputHiddenNeuron(
                targetList[dataIndex],
                outputValuePerData,
                hiddenLayer[dataIndex]
            ).forEach {
                weightCorrectionHiddenOutputPerData.add(it)
            }

            weightCorrectionHiddenOutput.add(weightCorrectionHiddenOutputPerData)
        }

        val biasCorrectionHiddenOutput = mutableListOf<List<Double>>()
        outputLayer.forEachIndexed { dataIndex, outputLayerPerData ->
            biasCorrectionHiddenOutput.add(
                calculateBiasCorrectionOutputHiddenNeuron(
                    targetList[dataIndex],
                    outputLayerPerData
                )
            )
        }

        // hidden ~ input
        val weightCorrectionInputHidden = mutableListOf<List<List<Double>>>()
        signalHiddenLayer.forEachIndexed { dataIndex, signalHidden ->
            val weightCorrectionInputHiddenPerData = mutableListOf<List<Double>>()
            calculateWeightCorrectionHiddenInputNeuron(
                targetList[dataIndex],
                outputLayer[dataIndex],
                signalHidden,
                inputLayer[dataIndex]
            ).forEach { weightListPerData ->
                weightCorrectionInputHiddenPerData.add(weightListPerData)
            }

            weightCorrectionInputHidden.add(
                weightCorrectionInputHiddenPerData
            )
        }

        val biasCorrectionInputHidden = mutableListOf<List<List<Double>>>()
        signalHiddenLayer.forEachIndexed { dataIndex, signalHidden ->
            val biasCorrectionInputHiddenPerData = mutableListOf<List<Double>>()
            calculateBiasCorrectionHiddenInputNeuron(
                targetList[dataIndex],
                outputLayer[dataIndex],
                signalHidden
            ).forEach {
                biasCorrectionInputHiddenPerData.add(it)
            }

            biasCorrectionInputHidden.add(biasCorrectionInputHiddenPerData)
        }

        // create jacobian
        val jacobianMatrix = getMatrixJacobian(
            weightCorrectionHiddenOutput,
            biasCorrectionHiddenOutput,
            weightCorrectionInputHidden,
            biasCorrectionInputHidden
        )

        // calculate Δw and Δv for update weight
        val hessianMatrix = createHessianMatrix(jacobianMatrix)
        val correctionAllWeightAndBias = calculateDeltaWeight(
            calculatePseudoInverse(
                hessianMatrix,
                Multik.identity(hessianMatrix.shape.first()),
                miu
            ),
            createGradientMatrix(
                jacobianMatrix,
                errorList
            )
        )

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
        val updatedInputHiddenWeight = mutableListOf<MutableList<Double>>()
        val updatedInputHiddenBias = mutableListOf<Double>()
        updateWeightData(
            inputWeight,
            correctedInputHiddenWeight
        ).forEach { row ->
            val weightValues = mutableListOf<Double>()
            row.forEachIndexed { columnIndex, value ->
                if (columnIndex == row.size - 1) {
                    updatedInputHiddenBias.add(value)
                } else {
                    weightValues.add(value)
                }
            }
            updatedInputHiddenWeight.add(weightValues)
        }

        val updatedHiddenOutputWeight = mutableListOf<MutableList<Double>>()
        val updatedHiddenOutputBias = mutableListOf<Double>()
        updateWeightData(
            hiddenWeight,
            correctedHiddenOutputWeight
        ).forEach { row ->
            val weightValues = mutableListOf<Double>()
            row.forEachIndexed { columnIndex, value ->
                if (columnIndex == row.size - 1) {
                    updatedHiddenOutputBias.add(value)
                } else {
                    weightValues.add(value)
                }
            }
            updatedHiddenOutputWeight.add(weightValues)
        }

        // return weight result
        return WeightResult(
            updatedInputHiddenWeight,
            updatedInputHiddenBias,
            updatedHiddenOutputWeight,
            updatedHiddenOutputBias
        )
    }

    /**
     * Menghitung koreksi bobot pada setiap neuron di output layer
     * @param targetValue as t: nilai target sesuai data input
     * @param outputLayer as y_k: nilai aktivasi output layer dengan tipe data double
     * @param hiddenLayer as z_j: nilai dari neuron hidden layer
     * @return (y_k - 1) * z_j, k == l || y_k * z_j, k != l
     */
    private fun calculateWeightCorrectionOutputHiddenNeuron(
        targetValue: Double,
        outputLayer: List<Double>,
        hiddenLayer: List<Double>
    ): List<List<Double>> {
        val weightCorrection = mutableListOf<List<Double>>()

        outputLayer.forEachIndexed { kIndex, outputValue ->
            val outputRowCorrection = mutableListOf<Double>()

            hiddenLayer.forEach { hiddenValue ->
                // (y_k - 1) * z_j, k == l
                // y_k * z_j, k != l
                outputRowCorrection.add(
                    if (targetValue.toInt() == kIndex) (outputValue - 1) * hiddenValue
                    else outputValue * hiddenValue
                )
            }

            weightCorrection.add(outputRowCorrection)
        }

        return weightCorrection
    }

    /**
     * Menghitung koreksi bias pada setiap neuron di output layer
     * @param targetValue as t: nilai target sesuai data input
     * @param outputLayer as y_k: nilai aktivasi output layer dengan tipe data double
     * @return y_k - 1, k == l || y_k, k != l
     */
    private fun calculateBiasCorrectionOutputHiddenNeuron(
        targetValue: Double,
        outputLayer: List<Double>
    ): List<Double> {
        val biasCorrection = mutableListOf<Double>()

        outputLayer.forEachIndexed { kIndex, outputValue ->
            // y_k - 1, k == l
            // y_k, k != l
            biasCorrection.add(
                if (targetValue.toInt() == kIndex) (outputValue - 1)
                else outputValue
            )
        }

        return biasCorrection
    }

    /**
     * Menghitung koreksi bobot pada setiap neuron di hidden layer
     * @param targetValue as t: nilai target sesuai data input
     * @param outputLayer as y_k: nilai aktivasi output layer dengan tipe data double
     * @param hiddenNetLayer as z_in_j: sinyal input dari input ke hidden layer dengan tipe data double
     * @param inputLayer as x_i: nilai neuron dari input layer
     * @return y_l * w_jl * f'(z_in_j) * x_i
     */
    private fun calculateWeightCorrectionHiddenInputNeuron(
        targetValue: Double,
        outputLayer: List<Double>,
        hiddenNetLayer: List<Double>,
        inputLayer: List<Double>
    ): List<List<Double>> {
        val weightCorrection = mutableListOf<MutableList<Double>>()

        outputLayer.forEachIndexed { kIndex, outputValue ->
            if (targetValue.toInt() == kIndex) {
                hiddenNetLayer.forEachIndexed { jIndex, hiddenNet ->
                    val hiddenRowCorrection = mutableListOf<Double>()
                    inputLayer.forEach { inputValue ->
                        // y_l * w_jl * f'(z_in_j) * x_i
                        val weight = hiddenWeight[kIndex][jIndex]
                        hiddenRowCorrection.add(
                            outputValue * weight * calculateDerivativeSigmoidFunction(
                                hiddenNet
                            ) * inputValue
                        )
                    }

                    weightCorrection.add(hiddenRowCorrection)
                }
            }
        }

        return weightCorrection
    }

    /**
     * Menghitung koreksi bias pada setiap neuron di output layer
     * @param targetValue as t: nilai target sesuai data input
     * @param outputLayer as y_k: nilai aktivasi output layer dengan tipe data double
     * @param hiddenNetLayer as z_in_j: sinyal input dari input ke hidden layer dengan tipe data double
     * @return f'(y_in_k) * f'(z_in_j)
     */
    private fun calculateBiasCorrectionHiddenInputNeuron(
        targetValue: Double,
        outputLayer: List<Double>,
        hiddenNetLayer: List<Double>
    ): List<List<Double>> {
        val biasCorrection = mutableListOf<List<Double>>()

        outputLayer.forEachIndexed { kIndex, outputValue ->
            val outputRowCorrection = mutableListOf<Double>()

            if (targetValue.toInt() == kIndex) {
                hiddenNetLayer.forEachIndexed { jIndex, hiddenNet ->
                    // y_l * w_jl * f'(z_in_j)
                    val weight = hiddenWeight[kIndex][jIndex]
                    outputRowCorrection.add(
                        outputValue * weight * calculateDerivativeSigmoidFunction(hiddenNet)
                    )
                }

                biasCorrection.add(outputRowCorrection)
            }
        }

        return biasCorrection
    }

    /**
     * Membentuk matriks jacobian
     * @param weightCorrectionHiddenOutputLayer as Δw_jk: koreksi bobot hidden ke output
     * @param biasCorrectionHiddenOutputLayer as Δb2_k: koreksi bias hidden ke output
     * @param weightCorrectionInputHiddenLayer as Δv_jk: koreksi bobot input ke hidden
     * @param biasCorrectionInputHiddenLayer as Δb1_j: koreksi bias input ke hidden
     * @return matrixJ
     */
    private fun getMatrixJacobian(
        weightCorrectionHiddenOutputLayer: List<List<List<Double>>>,
        biasCorrectionHiddenOutputLayer: List<List<Double>>,
        weightCorrectionInputHiddenLayer: List<List<List<Double>>>,
        biasCorrectionInputHiddenLayer: List<List<List<Double>>>
    ): List<List<Double>> {
        val matrixJ = mutableListOf<MutableList<Double>>()

        // add correction input ~ hidden
        weightCorrectionInputHiddenLayer.forEach { weightPerData ->
            val matrixJRow = mutableListOf<Double>()
            weightPerData.forEach { weightRow ->
                weightRow.forEach { weightValue ->
                    matrixJRow.add(weightValue)
                }
            }
            matrixJ.add(matrixJRow)
        }

        // add bias correction input ~ hidden
        matrixJ.forEachIndexed { index, matrixJRow ->
            biasCorrectionInputHiddenLayer[index].forEachIndexed { dataIndex, biasRowPerData ->
                biasRowPerData.forEach { biasRow ->
                    matrixJRow.add(biasRow)
                }
            }
        }

        // add weight correction hidden ~ output
        matrixJ.forEachIndexed { index, matrixJRow ->
            weightCorrectionHiddenOutputLayer[index].forEach { weightRow ->
                weightRow.forEach { weightValue ->
                    matrixJRow.add(weightValue)
                }
            }
        }

        // add bias correction hidden ~ output
        matrixJ.forEachIndexed { index, matrixJRow ->
            biasCorrectionHiddenOutputLayer[index].forEach { biasRow ->
                matrixJRow.add(biasRow)
            }
        }

        return matrixJ
    }

    /**
     * Menghitung matriks hessian
     * @param jacobianMatrix = matriks jacobian m x n dengan tipe data double
     * @return matrixH = transposeMatrix(jacobianMatrix) * jacobianMatrix
     */
    private fun createHessianMatrix(jacobianMatrix: List<List<Double>>): NDArray<Double, D2> {
        val transposedJacobian = Matrix.transposeMatrix(jacobianMatrix).toNDArray()
        return transposedJacobian.dot(jacobianMatrix.toNDArray())
    }

    /**
     * Menghitung matriks gradient
     * @param jacobianMatrix = matriks jacobian m x n dengan tipe data double
     * @param outputError as e = matriks error output jaringan m x 1 dengan tipe data double
     * @return matrixH = transposeMatrix(jacobianMatrix) * jacobianMatrix
     */
    private fun createGradientMatrix(
        jacobianMatrix: List<List<Double>>,
        outputError: List<Double>
    ): List<Double> {
        val transposedJacobian = Matrix.transposeMatrix(jacobianMatrix)

        return timesMatrixWithColumnMatrix(transposedJacobian, outputError)
    }

    /**
     * @param pseudoInverseMatrix = matriks pseudo inverse m x n dengan tipe data double
     * @param gradientMatrix as g = matriks gradient jaringan m x 1 dengan tipe data double
     * @return Δw = pseudoInverse * e
     */
    private fun calculateDeltaWeight(
        pseudoInverseMatrix: List<List<Double>>,
        gradientMatrix: List<Double>
    ): List<Double> {
        return timesMatrixWithColumnMatrix(pseudoInverseMatrix, gradientMatrix)
    }

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
        return (lastWeight.toNDArray() - deltaWeight.toNDArray()).toListD2()
    }
}