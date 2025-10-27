static int kalman_smoothen(WMAVoiceContext *s, int pitch,

                           const float *in, float *out, int size)

{

    int n;

    float optimal_gain = 0, dot;

    const float *ptr = &in[-FFMAX(s->min_pitch_val, pitch - 3)],

                *end = &in[-FFMIN(s->max_pitch_val, pitch + 3)],

                *best_hist_ptr;



    /* find best fitting point in history */

    do {

        dot = ff_scalarproduct_float_c(in, ptr, size);

        if (dot > optimal_gain) {

            optimal_gain  = dot;

            best_hist_ptr = ptr;

        }

    } while (--ptr >= end);



    if (optimal_gain <= 0)

        return -1;

    dot = ff_scalarproduct_float_c(best_hist_ptr, best_hist_ptr, size);

    if (dot <= 0) // would be 1.0

        return -1;



    if (optimal_gain <= dot) {

        dot = dot / (dot + 0.6 * optimal_gain); // 0.625-1.000

    } else

        dot = 0.625;



    /* actual smoothing */

    for (n = 0; n < size; n++)

        out[n] = best_hist_ptr[n] + dot * (in[n] - best_hist_ptr[n]);



    return 0;

}
