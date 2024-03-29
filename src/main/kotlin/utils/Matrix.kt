package utils

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import java.io.IOException

object Matrix {

    /**
     * Membaca data dari file csv
     * @return matrix m x n
     */
    fun readCsvFile(): List<List<Double>> {
        val data = mutableListOf<MutableList<Double>>()

        csvReader().open("src/data.csv") {
            readAllWithHeaderAsSequence().forEach {
                // add data to record
                val record = mutableListOf<Double>()
                it.values.forEach { value ->
                    record.add(value.toDouble())
                }

                data.add(record)
            }
        }

        return data
    }

    /**
     * Memberikan output dari array ke layar pengguna
     * @param matrix = array m x n dengan tipe data double
     */
    fun printMatrix(matrix: List<List<Double>>) {
        matrix.forEach { record ->
            print("|")

            record.forEachIndexed { index, recordData ->
                if (index != record.size - 1) print("$recordData ")
                else print(recordData)
            }

            println("|")
        }
    }

    /**
     * Mencari matrix identitas sesuai dimensi
     * @param dimension = jumlah dimensi dari matrix
     * @return matrix m x m
     */
    fun calculateMatrixIdentity(dimension: Int): List<List<Double>> {
        val data = mutableListOf<List<Double>>()

        for (i in 0..dimension) {
            val record = mutableListOf<Double>()

            for (j in 0..dimension) {
                if (i == j) record.add(1.0)
                else record.add(0.0)
            }

            data.add(record)
        }

        return data
    }

    /**
     * Melakukan operasi perkalian konstanta dan matriks
     * @param constants = konstanta dengan tipe data double
     * @param matrixA = array m x n dengan tipe data double
     * @return hasil perkalian dari constanta dan matrixA
     */
    fun timesConstWithMatrix(constants: Double, matrixA: List<List<Double>>): List<List<Double>> {
        val matrixResult = mutableListOf<List<Double>>()

        matrixA.forEach { row ->
            val rowMatrixResult = mutableListOf<Double>()
            row.forEach { value ->
                rowMatrixResult.add(constants * value)
            }

            matrixResult.add(rowMatrixResult)
        }

        return matrixResult
    }

    /**
     * Melakukan operasi perkalian dua buah matriks persegi
     * @param matrixA = array m x m dengan tipe data double
     * @param matrixB = array m x m dengan tipe data double
     * @return hasil perkalian dari matrixA dan matrixB
     */
    fun timesSquareMatrix(
        matrixA: List<List<Double>>,
        matrixB: List<List<Double>>
    ): List<List<Double>> {
        val matrixResult = mutableListOf<List<Double>>()
        val rowSize = matrixA.size
        val columnSize = matrixA.first().size

        for (row in 0 until rowSize) {
            val rowMatrixResult = mutableListOf<Double>()

            for (column in 0 until columnSize) {
                var sum = 0.0

                for (timesColumn in 0 until columnSize) {
                    sum += (matrixA[row][timesColumn] * matrixB[timesColumn][column])
                }

                rowMatrixResult.add(sum)
            }
            matrixResult.add(rowMatrixResult)
        }

        return matrixResult
    }

    /**
     * Melakukan operasi perkalian dua buah matriks
     * @param matrixA = array k x l dengan tipe data double
     * @param matrixB = array l x m dengan tipe data double
     * @return hasil perkalian dari matrixA dan matrixB
     */
    fun timesNonSquareMatrix(
        matrixA: List<List<Double>>,
        matrixB: List<List<Double>>
    ): List<List<Double>> {
        val rowMatrixASize = matrixA.size
        val columnMatrixASize = matrixA.first().size
        val rowMatrixBSize = matrixB.size
        val columnMatrixBSize = matrixB.first().size

        // check dimensions of two matrix
        if (columnMatrixASize != rowMatrixBSize) throw IOException()

        // continue if the dimensions is valid
        val resultMatrix = mutableListOf<List<Double>>()

        for (row in 0 until rowMatrixASize) {
            val rowMatrixResult = mutableListOf<Double>()

            for (columnB in 0 until columnMatrixBSize) {
                var sum = 0.0

                for (column in 0 until columnMatrixASize) {
                    sum += (matrixA[row][column] * matrixB[column][columnB])
                }

                rowMatrixResult.add(sum)
            }

            resultMatrix.add(rowMatrixResult)
        }

        return resultMatrix
    }

    /**
     * Melakukan operasi perkalian dua buah matriks
     * @param matrixA = matriks m x n dengan tipe data double
     * @param matrixB = matriks n x 1 dengan tipe data double
     * @return hasil perkalian dari matrixA dan matrixB
     */
    fun timesMatrixWithColumnMatrix(
        matrixA: List<List<Double>>,
        matrixB: List<Double>
    ): List<Double> {
        val rowMatrixASize = matrixA.size
        val columnMatrixASize = matrixA.first().size
        val rowMatrixBSize = matrixB.size

        // check dimensions of two matrix
        if (columnMatrixASize != rowMatrixBSize) {
            println("columnMatrixASize: $columnMatrixASize")
            println("rowMatrixBSize: $rowMatrixBSize")
            throw IOException()
        }

        // continue if the dimensions is valid
        val resultMatrix = mutableListOf<Double>()

        for (row in 0 until rowMatrixASize) {
            var sum = 0.0

            for (column in 0 until columnMatrixASize) {
                sum += (matrixA[row][column] * matrixB[column])
            }
            resultMatrix.add(sum)
        }

        return resultMatrix
    }

    /**
     * Melakukan operasi penjumlahan dua buah matriks
     * @param matrixA = array m x n dengan tipe data double
     * @param matrixB = array m x n dengan tipe data double
     * @return hasil penjumlahan dari matrixA dan matrixB
     */
    fun sumTwoMatrix(
        matrixA: List<List<Double>>,
        matrixB: List<List<Double>>
    ): List<List<Double>> {
        val totalMatrix = mutableListOf<List<Double>>()

        matrixA.forEachIndexed { rowIndex, rows ->
            val newRow = mutableListOf<Double>()

            rows.forEachIndexed { index, value ->
                newRow.add(value + matrixB[rowIndex][index])
            }

            totalMatrix.add(newRow)
        }

        return totalMatrix
    }

    /**
     * Melakukan operasi transpose dari @param matrix
     * @param matrix = array m x n dengan tipe data double
     */
    fun transposeMatrix(matrix: List<List<Double>>): List<List<Double>> {
        val rowMatrix = matrix.size
        val columnMatrix = matrix.first().size

        val transposedMatrix = mutableListOf<List<Double>>()
        for (i in 0 until columnMatrix) {
            val rowTransposedMatrix = mutableListOf<Double>()
            for (j in 0 until rowMatrix) {
                rowTransposedMatrix.add(matrix[j][i])
            }

            transposedMatrix.add(rowTransposedMatrix)
        }

        return transposedMatrix
    }

    /**
     * Menghitung matriks hessian
     * @param jacobianMatrix = matriks jacobian m x n dengan tipe data double
     * @return matrixH = transposeMatrix(jacobianMatrix) * jacobianMatrix
     */
    fun createHessianMatrix(jacobianMatrix: List<List<Double>>): List<List<Double>> {
        val transposedJacobian = transposeMatrix(jacobianMatrix)

        return timesNonSquareMatrix(transposedJacobian, jacobianMatrix)
    }

    /**
     * Menghitung matriks hessian
     * @param hessianMatrix = matriks hessian m x m dengan tipe data double
     * @param jacobianMatrix = matriks jacobian m x n dengan tipe data double
     * @param identityMatrix = matriks identitas m x m dengan tipe data double
     * @param miu = konstanta dengan tipe data double
     * @return pseudoInverse = inverse(J * transpose(J) + miu * I) * transpose(J)
     */
    fun calculatePseudoInverse(
        hessianMatrix: NDArray<Double, D2>,
        identityMatrix: NDArray<Double, D2>,
        miu: Double,
    ): List<List<Double>> {
        val sumMatrix = hessianMatrix + (identityMatrix * miu)
        val inverseMatrix = mk.linalg.inv(sumMatrix)

        return inverseMatrix.toListD2()
    }

}