int ff_unlock_avcodec(const AVCodec *codec)

{

    _Bool exp = 1;

    if (codec->caps_internal & FF_CODEC_CAP_INIT_THREADSAFE || !codec->init)

        return 0;



    av_assert0(atomic_compare_exchange_strong(&ff_avcodec_locked, &exp, 0));

    atomic_fetch_add(&entangled_thread_counter, -1);

    if (lockmgr_cb) {

        if ((*lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE))

            return -1;

    }



    return 0;

}
