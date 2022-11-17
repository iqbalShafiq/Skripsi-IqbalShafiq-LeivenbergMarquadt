import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.google.gson.JsonObject
import utils.Matrix

@OptIn(ExperimentalStdlibApi::class)
suspend fun main() {
    Matrix.readCsvFile()
}