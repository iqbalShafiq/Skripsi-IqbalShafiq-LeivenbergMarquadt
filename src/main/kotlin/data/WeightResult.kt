package data

data class WeightResult(
    val inputHiddenWeight: MutableList<MutableList<Double>>,
    val inputHiddenBias: MutableList<Double>,
    val hiddenOutputWeight: MutableList<MutableList<Double>>,
    val hiddenOutputBias: MutableList<Double>
)
