static av_cold int mdct_init(AVCodecContext *avctx, AC3MDCTContext *mdct,

                             int nbits)

{

    int i, n, n4, ret;



    n  = 1 << nbits;

    n4 = n >> 2;



    mdct->nbits = nbits;



    ret = fft_init(avctx, mdct, nbits - 2);

    if (ret)

        return ret;



    mdct->window = ff_ac3_window;



    FF_ALLOC_OR_GOTO(avctx, mdct->xcos1,    n4 * sizeof(*mdct->xcos1),    mdct_alloc_fail);

    FF_ALLOC_OR_GOTO(avctx, mdct->xsin1,    n4 * sizeof(*mdct->xsin1),    mdct_alloc_fail);

    FF_ALLOC_OR_GOTO(avctx, mdct->rot_tmp,  n  * sizeof(*mdct->rot_tmp),  mdct_alloc_fail);

    FF_ALLOC_OR_GOTO(avctx, mdct->cplx_tmp, n4 * sizeof(*mdct->cplx_tmp), mdct_alloc_fail);



    for (i = 0; i < n4; i++) {

        float alpha = 2.0 * M_PI * (i + 1.0 / 8.0) / n;

        mdct->xcos1[i] = FIX15(-cos(alpha));

        mdct->xsin1[i] = FIX15(-sin(alpha));

    }



    return 0;

mdct_alloc_fail:

    mdct_end(mdct);

    return AVERROR(ENOMEM);

}
