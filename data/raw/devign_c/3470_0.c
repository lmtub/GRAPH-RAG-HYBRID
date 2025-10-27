static int frame_thread_init(AVCodecContext *avctx)

{

    int thread_count = avctx->thread_count;

    AVCodec *codec = avctx->codec;

    AVCodecContext *src = avctx;

    FrameThreadContext *fctx;

    int i, err = 0;



    if (!thread_count) {

        int nb_cpus = get_logical_cpus(avctx);

        // use number of cores + 1 as thread count if there is motre than one

        if (nb_cpus > 1)

            thread_count = avctx->thread_count = nb_cpus + 1;

    }



    if (thread_count <= 1) {

        avctx->active_thread_type = 0;

        return 0;

    }



    avctx->thread_opaque = fctx = av_mallocz(sizeof(FrameThreadContext));



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



        p->parent = fctx;

        p->avctx  = copy;



        if (!copy) {

            err = AVERROR(ENOMEM);

            goto error;

        }



        *copy = *src;

        copy->thread_opaque = p;

        copy->pkt = &p->avpkt;



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

            copy->internal = av_malloc(sizeof(AVCodecInternal));

            if (!copy->internal) {

                err = AVERROR(ENOMEM);

                goto error;

            }

            *(copy->internal) = *(src->internal);

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

    frame_thread_free(avctx, i+1);



    return err;

}
