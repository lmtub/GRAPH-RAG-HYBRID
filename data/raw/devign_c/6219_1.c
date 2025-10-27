static void frame_thread_free(AVCodecContext *avctx, int thread_count)

{

    FrameThreadContext *fctx = avctx->thread_opaque;

    AVCodec *codec = avctx->codec;

    int i;



    park_frame_worker_threads(fctx, thread_count);



    if (fctx->prev_thread && fctx->prev_thread != fctx->threads)

        update_context_from_thread(fctx->threads->avctx, fctx->prev_thread->avctx, 0);



    fctx->die = 1;



    for (i = 0; i < thread_count; i++) {

        PerThreadContext *p = &fctx->threads[i];



        pthread_mutex_lock(&p->mutex);

        pthread_cond_signal(&p->input_cond);

        pthread_mutex_unlock(&p->mutex);



        pthread_join(p->thread, NULL);



        if (codec->close)

            codec->close(p->avctx);



        avctx->codec = NULL;



        release_delayed_buffers(p);

    }



    for (i = 0; i < thread_count; i++) {

        PerThreadContext *p = &fctx->threads[i];



        avcodec_default_free_buffers(p->avctx);



        pthread_mutex_destroy(&p->mutex);

        pthread_mutex_destroy(&p->progress_mutex);

        pthread_cond_destroy(&p->input_cond);

        pthread_cond_destroy(&p->progress_cond);

        pthread_cond_destroy(&p->output_cond);

        av_freep(&p->avpkt.data);



        if (i)

            av_freep(&p->avctx->priv_data);



        av_freep(&p->avctx);

    }



    av_freep(&fctx->threads);

    pthread_mutex_destroy(&fctx->buffer_mutex);

    av_freep(&avctx->thread_opaque);

    avctx->has_b_frames -= avctx->thread_count - 1;

}
