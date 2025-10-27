void uninit_opts(void)

{

    int i;

    for (i = 0; i < AVMEDIA_TYPE_NB; i++)

        av_freep(&avcodec_opts[i]);

    av_freep(&avformat_opts->key);

    av_freep(&avformat_opts);

#if CONFIG_SWSCALE

    av_freep(&sws_opts);

#endif



}