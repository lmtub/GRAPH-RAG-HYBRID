int opt_default(void *optctx, const char *opt, const char *arg)

{

    const AVOption *o;

    int consumed = 0;

    char opt_stripped[128];

    const char *p;

    const AVClass *cc = avcodec_get_class(), *fc = avformat_get_class();

#if CONFIG_AVRESAMPLE

    const AVClass *rc = avresample_get_class();

#endif

    const AVClass *sc, *swr_class;



    if (!strcmp(opt, "debug") || !strcmp(opt, "fdebug"))

        av_log_set_level(AV_LOG_DEBUG);



    if (!(p = strchr(opt, ':')))

        p = opt + strlen(opt);

    av_strlcpy(opt_stripped, opt, FFMIN(sizeof(opt_stripped), p - opt + 1));



    if ((o = opt_find(&cc, opt_stripped, NULL, 0,

                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) ||

        ((opt[0] == 'v' || opt[0] == 'a' || opt[0] == 's') &&

         (o = opt_find(&cc, opt + 1, NULL, 0, AV_OPT_SEARCH_FAKE_OBJ)))) {

        av_dict_set(&codec_opts, opt, arg, FLAGS);

        consumed = 1;

    }

    if ((o = opt_find(&fc, opt, NULL, 0,

                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        av_dict_set(&format_opts, opt, arg, FLAGS);

        if (consumed)

            av_log(NULL, AV_LOG_VERBOSE, "Routing option %s to both codec and muxer layer\n", opt);

        consumed = 1;

    }

#if CONFIG_SWSCALE

    sc = sws_get_class();

    if (!consumed && (o = opt_find(&sc, opt, NULL, 0,

                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        struct SwsContext *sws = sws_alloc_context();

        int ret = av_opt_set(sws, opt, arg, 0);

        sws_freeContext(sws);

        if (!strcmp(opt, "srcw") || !strcmp(opt, "srch") ||

            !strcmp(opt, "dstw") || !strcmp(opt, "dsth") ||

            !strcmp(opt, "src_format") || !strcmp(opt, "dst_format")) {

            av_log(NULL, AV_LOG_ERROR, "Directly using swscale dimensions/format options is not supported, please use the -s or -pix_fmt options\n");

            return AVERROR(EINVAL);

        }

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);

            return ret;

        }



        av_dict_set(&sws_dict, opt, arg, FLAGS);



        consumed = 1;

    }

#else

    if (!consumed && !strcmp(opt, "sws_flags")) {

        av_log(NULL, AV_LOG_WARNING, "Ignoring %s %s, due to disabled swscale\n", opt, arg);

        consumed = 1;

    }

#endif

#if CONFIG_SWRESAMPLE

    swr_class = swr_get_class();

    if (!consumed && (o=opt_find(&swr_class, opt, NULL, 0,

                                    AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        struct SwrContext *swr = swr_alloc();

        int ret = av_opt_set(swr, opt, arg, 0);

        swr_free(&swr);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);

            return ret;

        }

        av_dict_set(&swr_opts, opt, arg, FLAGS);

        consumed = 1;

    }

#endif

#if CONFIG_AVRESAMPLE

    if ((o=opt_find(&rc, opt, NULL, 0,

                       AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        av_dict_set(&resample_opts, opt, arg, FLAGS);

        consumed = 1;

    }

#endif



    if (consumed)

        return 0;

    return AVERROR_OPTION_NOT_FOUND;

}
