package mpl

object EpochRegulator {

    // init all parameter
    private val miu = 0.1
    private val tau = 10.0
    private val errorTarget = 0.5
    private val maxEpoch = 2

    fun startEpoch(
        onEpochStarted: () -> Unit
    ) {
        for (currentEpoch in 0 .. maxEpoch) {
            onEpochStarted()
        }
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
        marquardt: Double,
        tau: Double
    ): Double {
        if (latestMSE <= currentMSE) {
            return marquardt / tau
        }

        return marquardt * tau
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