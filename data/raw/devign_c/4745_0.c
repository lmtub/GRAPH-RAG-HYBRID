static int a52_decode_init(AVCodecContext *avctx)

{

    AC3DecodeState *s = avctx->priv_data;



#ifdef CONFIG_LIBA52BIN

    s->handle = dlopen(liba52name, RTLD_LAZY);

    if (!s->handle)

    {

        av_log( avctx, AV_LOG_ERROR, "A52 library %s could not be opened! \n%s\n", liba52name, dlerror());

        return -1;

    }

    s->a52_init = (a52_state_t* (*)(uint32_t)) dlsymm(s->handle, "a52_init");

    s->a52_samples = (sample_t* (*)(a52_state_t*)) dlsymm(s->handle, "a52_samples");

    s->a52_syncinfo = (int (*)(uint8_t*, int*, int*, int*)) dlsymm(s->handle, "a52_syncinfo");

    s->a52_frame = (int (*)(a52_state_t*, uint8_t*, int*, sample_t*, sample_t)) dlsymm(s->handle, "a52_frame");

    s->a52_block = (int (*)(a52_state_t*)) dlsymm(s->handle, "a52_block");

    s->a52_free = (void (*)(a52_state_t*)) dlsymm(s->handle, "a52_free");

    if (!s->a52_init || !s->a52_samples || !s->a52_syncinfo

        || !s->a52_frame || !s->a52_block || !s->a52_free)

    {

        dlclose(s->handle);

        return -1;

    }

#else

    s->handle = 0;

    s->a52_init = a52_init;

    s->a52_samples = a52_samples;

    s->a52_syncinfo = a52_syncinfo;

    s->a52_frame = a52_frame;

    s->a52_block = a52_block;

    s->a52_free = a52_free;

#endif

    s->state = s->a52_init(0); /* later use CPU flags */

    s->samples = s->a52_samples(s->state);

    s->inbuf_ptr = s->inbuf;

    s->frame_size = 0;



    /* allow downmixing to stereo or mono */

    if (avctx->channels > 0 && avctx->request_channels > 0 &&

            avctx->request_channels < avctx->channels &&

            avctx->request_channels <= 2) {

        avctx->channels = avctx->request_channels;

    }



    return 0;

}
