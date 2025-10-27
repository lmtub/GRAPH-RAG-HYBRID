static int thread_execute(AVFilterContext *ctx, avfilter_action_func *func,

                          void *arg, int *ret, int nb_jobs)

{

    ThreadContext *c = ctx->graph->internal->thread;

    int dummy_ret;



    if (nb_jobs <= 0)

        return 0;



    pthread_mutex_lock(&c->current_job_lock);



    c->current_job = c->nb_threads;

    c->nb_jobs     = nb_jobs;

    c->ctx         = ctx;

    c->arg         = arg;

    c->func        = func;

    if (ret) {

        c->rets    = ret;

        c->nb_rets = nb_jobs;

    } else {

        c->rets    = &dummy_ret;

        c->nb_rets = 1;

    }

    c->current_execute++;



    pthread_cond_broadcast(&c->current_job_cond);



    slice_thread_park_workers(c);



    return 0;

}
