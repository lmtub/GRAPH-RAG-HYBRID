static AVFrame *do_vmaf(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)

{

    LIBVMAFContext *s = ctx->priv;



    pthread_mutex_lock(&s->lock);



    while (s->frame_set != 0) {

        pthread_cond_wait(&s->cond, &s->lock);

    }



    av_frame_ref(s->gref, ref);

    av_frame_ref(s->gmain, main);



    s->frame_set = 1;



    pthread_cond_signal(&s->cond);

    pthread_mutex_unlock(&s->lock);



    return main;

}
