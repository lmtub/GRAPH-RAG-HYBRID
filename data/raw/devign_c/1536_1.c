int opt_default(void *optctx, const char *opt, const char *arg)

{

    const AVOption *o;

    int consumed = 0;

    char opt_stripped[128];

    const char *p;

    const AVClass *cc = avcodec_get_class(), *fc = avformat_get_class();

    const av_unused AVClass *rc_class;

    const AVClass *sc, *swr_class;



    if (!strcmp(opt, "debug") || !strcmp(opt, "fdebug"))

        av_log_set_level(AV_LOG_DEBUG);



    if (!(p = strchr(opt, ':')))

        p = opt + strlen(opt);

    av_strlcpy(opt_stripped, opt, FFMIN(sizeof(opt_stripped), p - opt + 1));



    if ((o = av_opt_find(&cc, opt_stripped, NULL, 0,

                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) ||

        ((opt[0] == 'v' || opt[0] == 'a' || opt[0] == 's') &&

         (o = av_opt_find(&cc, opt + 1, NULL, 0, AV_OPT_SEARCH_FAKE_OBJ)))) {

        av_dict_set(&codec_opts, opt, arg, FLAGS);

        consumed = 1;

    }

    if ((o = av_opt_find(&fc, opt, NULL, 0,

                              AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        av_dict_set(&format_opts, opt, arg, FLAGS);

        if(consumed)

            av_log(NULL, AV_LOG_VERBOSE, "Routing %s to codec and muxer layer\n", opt);

        consumed = 1;

    }

#if CONFIG_SWSCALE

    sc = sws_get_class();

    if (!consumed && av_opt_find(&sc, opt, NULL, 0,

                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) {

        // XXX we only support sws_flags, not arbitrary sws options

        int ret = av_opt_set(sws_opts, opt, arg, 0);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);

            return ret;

        }

        consumed = 1;

    }

#endif

#if CONFIG_SWRESAMPLE

    swr_class = swr_get_class();

    if (!consumed && av_opt_find(&swr_class, opt, NULL, 0,

                               AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) {

        int ret = av_opt_set(swr_opts, opt, arg, 0);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);

            return ret;

        }

        consumed = 1;

    }

#endif

#if CONFIG_AVRESAMPLE

    rc_class = avresample_get_class();

    if (av_opt_find(&rc_class, opt, NULL, 0,

                    AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) {

        av_dict_set(&resample_opts, opt, arg, FLAGS);

        consumed = 1;

    }

#endif



    if (consumed)

        return 0;

    return AVERROR_OPTION_NOT_FOUND;

}
