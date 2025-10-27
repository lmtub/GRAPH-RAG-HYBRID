static av_cold int libgsm_close(AVCodecContext *avctx) {

    av_freep(&avctx->coded_frame);

    gsm_destroy(avctx->priv_data);

    avctx->priv_data = NULL;

    return 0;

}
