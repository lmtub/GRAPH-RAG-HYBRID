static int quantize_coefs(double *coef, int *idx, float *lpc, int order)

{

    int i;

    uint8_t u_coef;

    const float *quant_arr = tns_tmp2_map[TNS_Q_BITS == 4];

    const double iqfac_p = ((1 << (TNS_Q_BITS-1)) - 0.5)/(M_PI/2.0);

    const double iqfac_m = ((1 << (TNS_Q_BITS-1)) + 0.5)/(M_PI/2.0);

    for (i = 0; i < order; i++) {

        idx[i] = ceilf(asin(coef[i])*((coef[i] >= 0) ? iqfac_p : iqfac_m));

        u_coef = (idx[i])&(~(~0<<TNS_Q_BITS));

        lpc[i] = quant_arr[u_coef];

    }

    return order;

}
