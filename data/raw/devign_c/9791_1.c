av_cold int avcodec_close(AVCodecContext *avctx)

{

    /* If there is a user-supplied mutex locking routine, call it. */

    if (ff_lockmgr_cb) {

        if ((*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_OBTAIN))

            return -1;

    }



    entangled_thread_counter++;

    if (entangled_thread_counter != 1) {

        av_log(avctx, AV_LOG_ERROR, "insufficient thread locking around avcodec_open/close()\n");

        entangled_thread_counter--;

        return -1;

    }



    if (avcodec_is_open(avctx)) {

        if (HAVE_THREADS && avctx->internal->frame_thread_encoder && avctx->thread_count > 1) {

            entangled_thread_counter --;

            ff_frame_thread_encoder_free(avctx);

            entangled_thread_counter ++;

        }

        if (HAVE_THREADS && avctx->thread_opaque)

            ff_thread_free(avctx);

        if (avctx->codec && avctx->codec->close)

            avctx->codec->close(avctx);

        avcodec_default_free_buffers(avctx);

        avctx->coded_frame = NULL;

        avctx->internal->byte_buffer_size = 0;

        av_freep(&avctx->internal->byte_buffer);

        av_freep(&avctx->internal);


    }



    if (avctx->priv_data && avctx->codec && avctx->codec->priv_class)

        av_opt_free(avctx->priv_data);

    av_opt_free(avctx);

    av_freep(&avctx->priv_data);

    if (av_codec_is_encoder(avctx->codec))

        av_freep(&avctx->extradata);

    avctx->codec = NULL;

    avctx->active_thread_type = 0;

    entangled_thread_counter--;



    /* Release any user-supplied mutex. */

    if (ff_lockmgr_cb) {

        (*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE);

    }

    return 0;

}