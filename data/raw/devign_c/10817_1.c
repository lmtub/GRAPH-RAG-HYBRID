int avcodec_close(AVCodecContext *avctx)

{

    entangled_thread_counter++;

    if(entangled_thread_counter != 1){

        av_log(avctx, AV_LOG_ERROR, "insufficient thread locking around avcodec_open/close()\n");

        entangled_thread_counter--;

        return -1;

    }



    if (ENABLE_THREADS && avctx->thread_opaque)

        avcodec_thread_free(avctx);

    if (avctx->codec->close)

        avctx->codec->close(avctx);

    avcodec_default_free_buffers(avctx);

    av_freep(&avctx->priv_data);


    avctx->codec = NULL;

    entangled_thread_counter--;

    return 0;

}