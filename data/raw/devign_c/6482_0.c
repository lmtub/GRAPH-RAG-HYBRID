static void postfilter(EVRCContext *e, float *in, const float *coeff,

                       float *out, int idx, const struct PfCoeff *pfc,

                       int length)

{

    float wcoef1[FILTER_ORDER], wcoef2[FILTER_ORDER],

          scratch[SUBFRAME_SIZE], temp[SUBFRAME_SIZE],

          mem[SUBFRAME_SIZE];

    float sum1 = 0.0, sum2 = 0.0, gamma, gain;

    float tilt = pfc->tilt;

    int i, n, best;



    bandwidth_expansion(wcoef1, coeff, pfc->p1);

    bandwidth_expansion(wcoef2, coeff, pfc->p2);



    /* Tilt compensation filter, TIA/IS-127 5.9.1 */

    for (i = 0; i < length - 1; i++)

        sum2 += in[i] * in[i + 1];

    if (sum2 < 0.0)

        tilt = 0.0;



    for (i = 0; i < length; i++) {

        scratch[i] = in[i] - tilt * e->last;

        e->last = in[i];

    }



    /* Short term residual filter, TIA/IS-127 5.9.2 */

    residual_filter(&e->postfilter_residual[ACB_SIZE], scratch, wcoef1, e->postfilter_fir, length);



    /* Long term postfilter */

    best = idx;

    for (i = FFMIN(MIN_DELAY, idx - 3); i <= FFMAX(MAX_DELAY, idx + 3); i++) {

        for (n = ACB_SIZE, sum2 = 0; n < ACB_SIZE + length; n++)

            sum2 += e->postfilter_residual[n] * e->postfilter_residual[n - i];

        if (sum2 > sum1) {

            sum1 = sum2;

            best = i;

        }

    }



    for (i = ACB_SIZE, sum1 = 0; i < ACB_SIZE + length; i++)

        sum1 += e->postfilter_residual[i - best] * e->postfilter_residual[i - best];

    for (i = ACB_SIZE, sum2 = 0; i < ACB_SIZE + length; i++)

        sum2 += e->postfilter_residual[i] * e->postfilter_residual[i - best];



    if (sum2 * sum1 == 0 || e->bitrate == RATE_QUANT) {

        memcpy(temp, e->postfilter_residual + ACB_SIZE, length * sizeof(float));

    } else {

        gamma = sum2 / sum1;

        if (gamma < 0.5)

            memcpy(temp, e->postfilter_residual + ACB_SIZE, length * sizeof(float));

        else {

            gamma = FFMIN(gamma, 1.0);



            for (i = 0; i < length; i++) {

                temp[i] = e->postfilter_residual[ACB_SIZE + i] + gamma *

                    pfc->ltgain * e->postfilter_residual[ACB_SIZE + i - best];

            }

        }

    }



    memcpy(scratch, temp, length * sizeof(float));

    memcpy(mem, e->postfilter_iir, FILTER_ORDER * sizeof(float));

    synthesis_filter(scratch, wcoef2, mem, length, scratch);



    /* Gain computation, TIA/IS-127 5.9.4-2 */

    for (i = 0, sum1 = 0, sum2 = 0; i < length; i++) {

        sum1 += in[i] * in[i];

        sum2 += scratch[i] * scratch[i];

    }

    gain = sum2 ? sqrt(sum1 / sum2) : 1.0;



    for (i = 0; i < length; i++)

        temp[i] *= gain;



    /* Short term postfilter */

    synthesis_filter(temp, wcoef2, e->postfilter_iir, length, out);



    memcpy(e->postfilter_residual,

           e->postfilter_residual + length, ACB_SIZE * sizeof(float));

}
