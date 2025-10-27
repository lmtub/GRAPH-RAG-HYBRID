static int thread_get_buffer_internal(AVCodecContext *avctx, ThreadFrame *f, int flags)

{

    PerThreadContext *p = avctx->internal->thread_ctx;

    int err;



    f->owner = avctx;



    ff_init_buffer_info(avctx, f->f);



    if (!(avctx->active_thread_type & FF_THREAD_FRAME))

        return ff_get_buffer(avctx, f->f, flags);



    if (atomic_load(&p->state) != STATE_SETTING_UP &&

        (avctx->codec->update_thread_context || !THREAD_SAFE_CALLBACKS(avctx))) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() cannot be called after ff_thread_finish_setup()\n");

        return -1;

    }



    if (avctx->internal->allocate_progress) {

        atomic_int *progress;

        f->progress = av_buffer_alloc(2 * sizeof(*progress));

        if (!f->progress) {

            return AVERROR(ENOMEM);

        }

        progress = (atomic_int*)f->progress->data;



        atomic_init(&progress[0], -1);

        atomic_init(&progress[1], -1);

    }



    pthread_mutex_lock(&p->parent->buffer_mutex);

    if (avctx->thread_safe_callbacks ||

        avctx->get_buffer2 == avcodec_default_get_buffer2) {

        err = ff_get_buffer(avctx, f->f, flags);

    } else {

        pthread_mutex_lock(&p->progress_mutex);

        p->requested_frame = f->f;

        p->requested_flags = flags;

        atomic_store_explicit(&p->state, STATE_GET_BUFFER, memory_order_release);

        pthread_cond_broadcast(&p->progress_cond);



        while (atomic_load(&p->state) != STATE_SETTING_UP)

            pthread_cond_wait(&p->progress_cond, &p->progress_mutex);



        err = p->result;



        pthread_mutex_unlock(&p->progress_mutex);



    }

    if (!THREAD_SAFE_CALLBACKS(avctx) && !avctx->codec->update_thread_context)

        ff_thread_finish_setup(avctx);

    if (err)

        av_buffer_unref(&f->progress);



    pthread_mutex_unlock(&p->parent->buffer_mutex);



    return err;

}
