static av_cold void mdct_end(AC3MDCTContext *mdct)

{

    mdct->nbits = 0;

    av_freep(&mdct->costab);

    av_freep(&mdct->sintab);

    av_freep(&mdct->xcos1);

    av_freep(&mdct->xsin1);

    av_freep(&mdct->rot_tmp);

    av_freep(&mdct->cplx_tmp);

}
