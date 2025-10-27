static int thread_get_buffer_internal(AVCodecContext *avctx, ThreadFrame *f, int flags)

{

    PerThreadContext *p = avctx->thread_opaque;

    int err;



    f->owner = avctx;



    ff_init_buffer_info(avctx, f->f);



    if (!(avctx->active_thread_type & FF_THREAD_FRAME))

        return ff_get_buffer(avctx, f->f, flags);



    if (p->state != STATE_SETTING_UP &&

        (avctx->codec->update_thread_context || (!avctx->thread_safe_callbacks &&

                avctx->get_buffer != avcodec_default_get_buffer))) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() cannot be called after ff_thread_finish_setup()\n");

        return -1;

    }



    if (avctx->internal->allocate_progress) {

        int *progress;

        f->progress = av_buffer_alloc(2 * sizeof(int));

        if (!f->progress) {

            return AVERROR(ENOMEM);

        }

        progress = (int*)f->progress->data;



        progress[0] = progress[1] = -1;

    }



    pthread_mutex_lock(&p->parent->buffer_mutex);



    if (avctx->thread_safe_callbacks || (

#if FF_API_GET_BUFFER

        !avctx->get_buffer &&

#endif

        avctx->get_buffer2 == avcodec_default_get_buffer2)) {

        err = ff_get_buffer(avctx, f->f, flags);

    } else {

        pthread_mutex_lock(&p->progress_mutex);

        p->requested_frame = f->f;

        p->requested_flags = flags;

        p->state = STATE_GET_BUFFER;

        pthread_cond_broadcast(&p->progress_cond);



        while (p->state != STATE_SETTING_UP)

            pthread_cond_wait(&p->progress_cond, &p->progress_mutex);



        err = p->result;



        pthread_mutex_unlock(&p->progress_mutex);



        if (!avctx->codec->update_thread_context)

            ff_thread_finish_setup(avctx);

    }



    if (err)

        av_buffer_unref(&f->progress);



    pthread_mutex_unlock(&p->parent->buffer_mutex);



    return err;

}
