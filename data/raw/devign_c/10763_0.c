void ff_celp_lp_zero_synthesis_filterf(float *out,

                                       const float* filter_coeffs,

                                       const float* in,

                                       int buffer_length,

                                       int filter_length)

{

    int i,n;



    // Avoids a +1 in the inner loop.

    filter_length++;



    for (n = 0; n < buffer_length; n++) {

        out[n] = in[n];

        for (i = 1; i < filter_length; i++)

            out[n] += filter_coeffs[i-1] * in[n-i];

    }

}
