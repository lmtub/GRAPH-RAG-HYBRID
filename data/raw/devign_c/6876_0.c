static void hybrid4_8_12_cx(float (*in)[2], float (*out)[32][2], const float (*filter)[7][2], int N, int len)

{

    int i, j, ssb;



    for (i = 0; i < len; i++, in++) {

        for (ssb = 0; ssb < N; ssb++) {

            float sum_re = filter[ssb][6][0] * in[6][0], sum_im = filter[ssb][6][0] * in[6][1];

            for (j = 0; j < 6; j++) {

                float in0_re = in[j][0];

                float in0_im = in[j][1];

                float in1_re = in[12-j][0];

                float in1_im = in[12-j][1];

                sum_re += filter[ssb][j][0] * (in0_re + in1_re) - filter[ssb][j][1] * (in0_im - in1_im);

                sum_im += filter[ssb][j][0] * (in0_im + in1_im) + filter[ssb][j][1] * (in0_re - in1_re);

            }

            out[ssb][i][0] = sum_re;

            out[ssb][i][1] = sum_im;

        }

    }

}
