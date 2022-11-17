package utils

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader

object Matrix {

    /**
     * Membaca data dari file csv
     * @return matrix m x n
     */
    suspend fun readCsvFile(): List<List<Int>> {
        val data = mutableListOf<MutableList<Int>>()

        csvReader().openAsync("src/data.csv") {
            readAllWithHeaderAsSequence().forEach {
                // add data to record
                val record = mutableListOf<Int>()
                it.values.forEach { value ->
                    record.add(value.toInt())
                }

                data.add(record)
            }
        }

        return data
    }

    /**
     * Mencari matrix identitas sesuai dimensi
     * @param dimension = jumlah dimensi dari matrix
     * @return matrix m x m
     */
    fun getMatrixIdentity(dimension: Int): List<List<Int>> {
        val data = mutableListOf<List<Int>>()

        for (i in 0..dimension) {
            val record = mutableListOf<Int>()

            for (j in 0..dimension) {
                if (i == j) record.add(1)
                else record.add(0)
            }

            data.add(record)
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
     * Melakukan operasi perkalian dua buah matriks
     * @param matrixA = array m x n dengan tipe data double
     * @param matrixB = array m x n dengan tipe data double
     * @return hasil perkalian dari matrixA dan matrixB
     */
    fun timesTwoMatrix(matrixA: Array<Double>, matrixB: Array<Double>): Array<Double> {
        return arrayOf()
    }

    /**
     * Melakukan operasi penjumlahan dua buah matriks
     * @param matrixA = array m x n dengan tipe data double
     * @param matrixB = array m x n dengan tipe data double
     * @return hasil penjumlahan dari matrixA dan matrixB
     */
    fun sumTwoMatrix(matrixA: Array<Double>, matrixB: Array<Double>): Array<Double> {
        return arrayOf()
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
            val recordTransposedMatrix = mutableListOf<Double>()
            for (j in 0 until rowMatrix) {
                recordTransposedMatrix.add(matrix[j][i])
            }

            transposedMatrix.add(recordTransposedMatrix)
        }

        return transposedMatrix
    }

    /**
     * Menghitung matriks jacobian untuk perhitungan matriks Hessian
     * @return matrixJ = array m x n tipe data array
     */
    fun calculateMatrixJacobian(): Array<Double> {
        val matrixJ = arrayOf<Double>()

        return matrixJ
    }

    /**
     * Menghitung matriks hessian dengan formula:
     * @return matrixH = (transposeMatrix(matrixJ) * matrixJ) + (miu * I)
     */
    fun createHessianMatrix(): Array<Double> {
        val matrixH = arrayOf<Double>()

        return matrixH
    }

}