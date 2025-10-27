int ff_thread_get_buffer(AVCodecContext *avctx, AVFrame *f)

{

    PerThreadContext *p = avctx->thread_opaque;

    int *progress, err;



    f->owner = avctx;



    ff_init_buffer_info(avctx, f);



    if (!(avctx->active_thread_type&FF_THREAD_FRAME)) {

        f->thread_opaque = NULL;

        return avctx->get_buffer(avctx, f);

    }



    if (p->state != STATE_SETTING_UP &&

        (avctx->codec->update_thread_context || (!avctx->thread_safe_callbacks &&

                avctx->get_buffer != avcodec_default_get_buffer))) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() cannot be called after ff_thread_finish_setup()\n");

        return -1;

    }



    pthread_mutex_lock(&p->parent->buffer_mutex);

    f->thread_opaque = progress = allocate_progress(p);



    if (!progress) {

        pthread_mutex_unlock(&p->parent->buffer_mutex);

        return -1;

    }



    progress[0] =

    progress[1] = -1;



    if (avctx->thread_safe_callbacks ||

        avctx->get_buffer == avcodec_default_get_buffer) {

        err = avctx->get_buffer(avctx, f);

    } else {

        p->requested_frame = f;

        p->state = STATE_GET_BUFFER;

        pthread_mutex_lock(&p->progress_mutex);

        pthread_cond_signal(&p->progress_cond);



        while (p->state != STATE_SETTING_UP)

            pthread_cond_wait(&p->progress_cond, &p->progress_mutex);



        err = p->result;



        pthread_mutex_unlock(&p->progress_mutex);



        if (!avctx->codec->update_thread_context)

            ff_thread_finish_setup(avctx);

    }



    pthread_mutex_unlock(&p->parent->buffer_mutex);



    return err;

}
