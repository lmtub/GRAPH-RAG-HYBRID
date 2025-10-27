static void mdct512(AC3MDCTContext *mdct, float *out, float *in)

{

    mdct->fft.mdct_calc(&mdct->fft, out, in);

}
