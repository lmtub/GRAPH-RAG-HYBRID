static void sbr_qmf_analysis(DSPContext *dsp, FFTContext *mdct, const float *in, float *x,

                             float z[320], float W[2][32][32][2])

{

    int i, k;

    memcpy(W[0], W[1], sizeof(W[0]));

    memcpy(x    , x+1024, (320-32)*sizeof(x[0]));

    memcpy(x+288, in,         1024*sizeof(x[0]));

    for (i = 0; i < 32; i++) { // numTimeSlots*RATE = 16*2 as 960 sample frames

                               // are not supported

        dsp->vector_fmul_reverse(z, sbr_qmf_window_ds, x, 320);

        for (k = 0; k < 64; k++) {

            float f = z[k] + z[k + 64] + z[k + 128] + z[k + 192] + z[k + 256];

            z[k] = f;

        }

        //Shuffle to IMDCT

        z[64] = z[0];

        for (k = 1; k < 32; k++) {

            z[64+2*k-1] =  z[   k];

            z[64+2*k  ] = -z[64-k];

        }

        z[64+63] = z[32];



        mdct->imdct_half(mdct, z, z+64);

        for (k = 0; k < 32; k++) {

            W[1][i][k][0] = -z[63-k];

            W[1][i][k][1] = z[k];

        }

        x += 32;

    }

}
