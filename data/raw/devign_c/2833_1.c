int ff_slice_thread_init(AVCodecContext *avctx)

{

    int i;

    SliceThreadContext *c;

    int thread_count = avctx->thread_count;



#if HAVE_W32THREADS

    w32thread_init();

#endif



    // We cannot do this in the encoder init as the threads are created before

    if (av_codec_is_encoder(avctx->codec) &&

        avctx->codec_id == AV_CODEC_ID_MPEG1VIDEO &&

        avctx->height > 2800)

        thread_count = avctx->thread_count = 1;



    if (!thread_count) {

        int nb_cpus = av_cpu_count();

        if  (avctx->height)

            nb_cpus = FFMIN(nb_cpus, (avctx->height+15)/16);

        // use number of cores + 1 as thread count if there is more than one

        if (nb_cpus > 1)

            thread_count = avctx->thread_count = FFMIN(nb_cpus + 1, MAX_AUTO_THREADS);

        else

            thread_count = avctx->thread_count = 1;

    }



    if (thread_count <= 1) {

        avctx->active_thread_type = 0;

        return 0;

    }



    c = av_mallocz(sizeof(SliceThreadContext));

    if (!c)

        return -1;



    c->workers = av_mallocz_array(thread_count, sizeof(pthread_t));

    if (!c->workers) {

        av_free(c);

        return -1;

    }



    avctx->internal->thread_ctx = c;

    c->current_job = 0;

    c->job_count = 0;

    c->job_size = 0;

    c->done = 0;

    pthread_cond_init(&c->current_job_cond, NULL);

    pthread_cond_init(&c->last_job_cond, NULL);

    pthread_mutex_init(&c->current_job_lock, NULL);

    pthread_mutex_lock(&c->current_job_lock);

    for (i=0; i<thread_count; i++) {

        if(pthread_create(&c->workers[i], NULL, worker, avctx)) {

           avctx->thread_count = i;

           pthread_mutex_unlock(&c->current_job_lock);

           ff_thread_free(avctx);

           return -1;

        }

    }



    thread_park_workers(c, thread_count);



    avctx->execute = thread_execute;

    avctx->execute2 = thread_execute2;

    return 0;

}
