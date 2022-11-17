package mpl

import utils.Matrix

class FeedForward {

    /**
     * Melakukan normalisasi dataset dengan membagi setiap pixel dengan 255
     * @return matrix m x n
     */
    private suspend fun normalize(): List<List<Double>> {
        // read data from csv
        val data = Matrix.readCsvFile()

        val normalizedData = mutableListOf<List<Double>>()
        data.forEach { record ->
            // get last position
            val lastPosition = record.size - 1
            val newRecord = mutableListOf<Double>()

            // divide every pixel with 255
            record.forEachIndexed { index, pixelValue ->
                // exclude label
                if (index != lastPosition) {
                    newRecord.add((pixelValue / NORMALIZATION_DIVIDER).toDouble())
                }
            }

            // add new record into normalized data
            normalizedData.add(newRecord)
        }

        return normalizedData
    }

    private fun initWeightData() {

    }

    private fun timesWeightAndPixelData() {

    }

    private fun createHiddenLayer() {

    }

    companion object {
        private const val NORMALIZATION_DIVIDER = 255
    }
}