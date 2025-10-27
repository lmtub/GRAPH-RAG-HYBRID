static int thread_execute(AVCodecContext *avctx, action_func* func, void *arg, int *ret, int job_count, int job_size)

{

    SliceThreadContext *c = avctx->internal->thread_ctx;



    if (!(avctx->active_thread_type&FF_THREAD_SLICE) || avctx->thread_count <= 1)

        return avcodec_default_execute(avctx, func, arg, ret, job_count, job_size);



    if (job_count <= 0)

        return 0;



    pthread_mutex_lock(&c->current_job_lock);



    c->current_job = avctx->thread_count;

    c->job_count = job_count;

    c->job_size = job_size;

    c->args = arg;

    c->func = func;

    c->rets = ret;

    c->current_execute++;

    pthread_cond_broadcast(&c->current_job_cond);



    thread_park_workers(c, avctx->thread_count);



    return 0;

}
