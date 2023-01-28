package mpl

object EpochRegulator {

    // init all parameter
    private const val MAX_EPOCH = 20
    private const val ERROR_TARGET = 0.1
    private const val TAU_FACTOR = 10.0
    const val MIU = 0.01

    fun startEpoch(
        onEpochStarted: (maxEpoch: Int, errorTarget: Double) -> Unit
    ) {
        onEpochStarted(MAX_EPOCH, ERROR_TARGET)
    }

    /**
     * Mendapatkan parameter levenberg-marquardt (μ) baru pada akhir epoch
     * @param currentMSE: MSE lama
     * @param latestMSE: MSE baru
     * @param marquardt as μ: parameter levenberg-marquardt
     * @param tau as τ: faktor tau
     * @return μ baru berdasarkan kondisi MSE
     */
    fun getNewLMParameter(
        currentMSE: Double,
        latestMSE: Double,
        marquardt: Double
    ): Double {
        if (latestMSE <= currentMSE) {
            return marquardt / TAU_FACTOR
        }

        return marquardt * TAU_FACTOR
    }

    /**
     * Melakukan pengecekan berakhirnya epoch berdasarkan target error
     * @param errorTarget: target dari error yang telah ditentukan di awal program
     * @param error: error yang didapatkan dari epoch terakhir
     * @return bernilai true jika error < errorTarget
     */
    fun isEpochOverByError(
        errorTarget: Double,
        error: Double
    ): Boolean {
        return error < errorTarget
    }
}