static float voice_factor(float *p_vector, float p_gain,

                          float *f_vector, float f_gain,

                          CELPMContext *ctx)

{

    double p_ener = (double) ctx->dot_productf(p_vector, p_vector,

                                                          AMRWB_SFR_SIZE) *

                    p_gain * p_gain;

    double f_ener = (double) ctx->dot_productf(f_vector, f_vector,

                                                          AMRWB_SFR_SIZE) *

                    f_gain * f_gain;



    return (p_ener - f_ener) / (p_ener + f_ener);

}
