package data

data class FeedForwardResult(
    val inputLayer: InputData,
    val hiddenLayer: List<List<Double>>,
    val activatedHiddenLayer: List<List<Double>>,
    val outputLayer: List<List<Double>>,
    val inputWeight: List<List<Double>>,
    val hiddenWeight: List<List<Double>>,
    val activatedOutputLayer: List<List<Double>>,
    val errorList: List<Double>,
    val mse: Double,
    val miu: Double
)