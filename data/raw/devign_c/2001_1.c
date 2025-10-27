static void IMLT(float *pInput, float *pOutput, int odd_band)

{

    int     i;



    if (odd_band) {

        /**

        * Reverse the odd bands before IMDCT, this is an effect of the QMF transform

        * or it gives better compression to do it this way.

        * FIXME: It should be possible to handle this in ff_imdct_calc

        * for that to happen a modification of the prerotation step of

        * all SIMD code and C code is needed.

        * Or fix the functions before so they generate a pre reversed spectrum.

        */



        for (i=0; i<128; i++)

            FFSWAP(float, pInput[i], pInput[255-i]);

    }



    ff_imdct_calc(&mdct_ctx,pOutput,pInput);



    /* Perform windowing on the output. */

    dsp.vector_fmul(pOutput,mdct_window,512);



}
