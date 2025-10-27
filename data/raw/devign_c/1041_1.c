double ff_lpc_calc_ref_coefs_f(LPCContext *s, const float *samples, int len,

                               int order, double *ref)

{

    int i;

    double signal = 0.0f, avg_err = 0.0f;

    double autoc[MAX_LPC_ORDER+1] = {0}, error[MAX_LPC_ORDER+1] = {0};

    const double a = 0.5f, b = 1.0f - a;



    /* Apply windowing */

    for (i = 0; i < len; i++) {

        double weight = a - b*cos((2*M_PI*i)/(len - 1));

        s->windowed_samples[i] = weight*samples[i];

    }



    s->lpc_compute_autocorr(s->windowed_samples, len, order, autoc);

    signal = autoc[0];

    compute_ref_coefs(autoc, order, ref, error);

    for (i = 0; i < order; i++)

        avg_err = (avg_err + error[i])/2.0f;

    return signal/avg_err;

}
