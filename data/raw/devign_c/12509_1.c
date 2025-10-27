void avcodec_flush_buffers(AVCodecContext *avctx)

{

    if(HAVE_PTHREADS && avctx->active_thread_type&FF_THREAD_FRAME)

        ff_thread_flush(avctx);

    if(avctx->codec->flush)

        avctx->codec->flush(avctx);

}
