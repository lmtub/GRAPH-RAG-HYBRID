static int thread_execute2(AVCodecContext *avctx, action_func2* func2, void *arg, int *ret, int job_count)

{

    ThreadContext *c= avctx->thread_opaque;

    c->func2 = func2;

    return thread_execute(avctx, NULL, arg, ret, job_count, 0);

}
