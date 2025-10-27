static av_cold int vtenc_close(AVCodecContext *avctx)

{

    VTEncContext *vtctx = avctx->priv_data;



    if(!vtctx->session) return 0;



    VTCompressionSessionInvalidate(vtctx->session);

    pthread_cond_destroy(&vtctx->cv_sample_sent);

    pthread_mutex_destroy(&vtctx->lock);

    CFRelease(vtctx->session);

    vtctx->session = NULL;



    return 0;

}
