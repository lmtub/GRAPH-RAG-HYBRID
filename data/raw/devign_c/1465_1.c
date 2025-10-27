int ff_frame_thread_init(AVCodecContext *avctx)

{

    int thread_count = avctx->thread_count;

    const AVCodec *codec = avctx->codec;

    AVCodecContext *src = avctx;

    FrameThreadContext *fctx;

    int i, err = 0;



#if HAVE_W32THREADS

    w32thread_init();

#endif



    if (!thread_count) {

        int nb_cpus = av_cpu_count();

        av_log(avctx, AV_LOG_DEBUG, "detected %d logical cores\n", nb_cpus);

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



    avctx->internal->thread_ctx = fctx = av_mallocz(sizeof(FrameThreadContext));



    fctx->threads = av_mallocz(sizeof(PerThreadContext) * thread_count);

    pthread_mutex_init(&fctx->buffer_mutex, NULL);

    fctx->delaying = 1;



    for (i = 0; i < thread_count; i++) {

        AVCodecContext *copy = av_malloc(sizeof(AVCodecContext));

        PerThreadContext *p  = &fctx->threads[i];



        pthread_mutex_init(&p->mutex, NULL);

        pthread_mutex_init(&p->progress_mutex, NULL);

        pthread_cond_init(&p->input_cond, NULL);

        pthread_cond_init(&p->progress_cond, NULL);

        pthread_cond_init(&p->output_cond, NULL);



        p->frame = av_frame_alloc();

        if (!p->frame) {


            err = AVERROR(ENOMEM);

            goto error;

        }



        p->parent = fctx;

        p->avctx  = copy;



        if (!copy) {

            err = AVERROR(ENOMEM);

            goto error;

        }



        *copy = *src;



        copy->internal = av_malloc(sizeof(AVCodecInternal));

        if (!copy->internal) {

            err = AVERROR(ENOMEM);

            goto error;

        }

        *copy->internal = *src->internal;

        copy->internal->thread_ctx = p;

        copy->internal->pkt = &p->avpkt;



        if (!i) {

            src = copy;



            if (codec->init)

                err = codec->init(copy);



            update_context_from_thread(avctx, copy, 1);

        } else {

            copy->priv_data = av_malloc(codec->priv_data_size);

            if (!copy->priv_data) {

                err = AVERROR(ENOMEM);

                goto error;

            }

            memcpy(copy->priv_data, src->priv_data, codec->priv_data_size);

            copy->internal->is_copy = 1;



            if (codec->init_thread_copy)

                err = codec->init_thread_copy(copy);

        }



        if (err) goto error;



        if (!pthread_create(&p->thread, NULL, frame_worker_thread, p))

            p->thread_init = 1;

    }



    return 0;



error:

    ff_frame_thread_free(avctx, i+1);



    return err;

}