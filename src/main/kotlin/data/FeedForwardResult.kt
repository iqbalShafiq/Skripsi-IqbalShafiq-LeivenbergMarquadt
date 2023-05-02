package data

data class FeedForwardResult(
    var targetList: List<Double>, // t
    var inputLayer: List<List<Double>>, // x
    var signalHiddenLayer: List<List<Double>>, // z_in
    var hiddenLayer: List<List<Double>>, // z
    var signalOutputLayer: List<List<Double>>, // y_in
    var outputLayer: List<List<Double>>, // y
    var errorList: List<Double>, // e_p
    var errorValue: Double,
    var accuracyValue: Double,
    var mse: Double,
)
