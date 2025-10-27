static void release_delayed_buffers(PerThreadContext *p)

{

    FrameThreadContext *fctx = p->parent;



    while (p->num_released_buffers > 0) {

        AVFrame *f = &p->released_buffers[--p->num_released_buffers];



        pthread_mutex_lock(&fctx->buffer_mutex);

        free_progress(f);

        f->thread_opaque = NULL;



        f->owner->release_buffer(f->owner, f);

        pthread_mutex_unlock(&fctx->buffer_mutex);

    }

}
