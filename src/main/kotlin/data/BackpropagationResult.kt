package data

data class BackpropagationResult(
    val inputHiddenWeight: List<List<Double>>,
    val hiddenOutputWeight: List<List<Double>>
)
