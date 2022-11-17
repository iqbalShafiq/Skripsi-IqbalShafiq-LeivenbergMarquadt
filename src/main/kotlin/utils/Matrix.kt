package utils

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.google.gson.JsonObject

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
            println(data)
        }

        return data
    }

    /**
     * Mengembalikan nilai mengembalikan dalam bentuk array of double
     * @param csvList = list of json object yang berisi data input data csv
     * @return matrix m x n
     */
    suspend fun getMatrix(csvList: List<JsonObject>): List<JsonObject> {
        val data = mutableListOf<JsonObject>()

        csvReader().openAsync("src/hmnist_28_28_L.csv") {
            readAllWithHeaderAsSequence().forEach {
                //Do something
                val jsonObject = JsonObject()
                it.keys.forEach { key ->
                    jsonObject.addProperty(key, it[key])
                }
                data.add(jsonObject)
            }
            println(data)
        }

        return data
    }

    /**
     * Memberikan output dari array ke layar pengguna
     * @param matrix = array m x n dengan tipe data double
     */
    fun printMatrix(matrix: Array<Double>) {

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
     * Mengembalikan matriks identitas dalam bentuk array n x n dengan tipe data double
     * @param n = dimensi matriks dengan tipe data integer
     * @return matriks identitas
     */
    fun calculateMatrixIdentity(n: Int): Array<Double> {
        return arrayOf()
    }

    /**
     * Melakukan operasi transpose dari @param matrix
     * @param matrix = array m x n dengan tipe data double
     */
    fun transposeMatrix(matrix: Array<Double>): Array<Double> {
        return arrayOf()
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