static int ffserver_save_avoption(const char *opt, const char *arg, int type, FFServerConfig *config)

{

    static int hinted = 0;

    int ret = 0;

    AVDictionaryEntry *e;

    const AVOption *o = NULL;

    const char *option = NULL;

    const char *codec_name = NULL;

    char buff[1024];

    AVCodecContext *ctx;

    AVDictionary **dict;

    enum AVCodecID guessed_codec_id;



    switch (type) {

    case AV_OPT_FLAG_VIDEO_PARAM:

        ctx = config->dummy_vctx;

        dict = &config->video_opts;

        guessed_codec_id = config->guessed_video_codec_id != AV_CODEC_ID_NONE ?

                           config->guessed_video_codec_id : AV_CODEC_ID_H264;

        break;

    case AV_OPT_FLAG_AUDIO_PARAM:

        ctx = config->dummy_actx;

        dict = &config->audio_opts;

        guessed_codec_id = config->guessed_audio_codec_id != AV_CODEC_ID_NONE ?

                           config->guessed_audio_codec_id : AV_CODEC_ID_AAC;

        break;

    default:

        av_assert0(0);

    }



    if (strchr(opt, ':')) {

        //explicit private option

        snprintf(buff, sizeof(buff), "%s", opt);

        codec_name = buff;

        option = strchr(buff, ':');

        buff[option - buff] = '\0';

        option++;

        if ((ret = ffserver_set_codec(ctx, codec_name, config)) < 0)

            return ret;

        if (!ctx->codec || !ctx->priv_data)

            return -1;

    } else {

        option = opt;

    }



    o = av_opt_find(ctx, option, NULL, type | AV_OPT_FLAG_ENCODING_PARAM, AV_OPT_SEARCH_CHILDREN);

    if (!o && (!strcmp(option, "time_base")  || !strcmp(option, "pixel_format") ||

               !strcmp(option, "video_size") || !strcmp(option, "codec_tag")))

        o = av_opt_find(ctx, option, NULL, 0, 0);

    if (!o) {

        report_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                            &config->errors, "Option not found: %s\n", opt);

        if (!hinted && ctx->codec_id == AV_CODEC_ID_NONE) {

            hinted = 1;

            report_config_error(config->filename, config->line_num, AV_LOG_ERROR, NULL,

                                "If '%s' is a codec private option, then prefix it with codec name, "

                                "for example '%s:%s %s' or define codec earlier.\n",

                                opt, avcodec_get_name(guessed_codec_id) ,opt, arg);

        }

    } else if ((ret = av_opt_set(ctx, option, arg, AV_OPT_SEARCH_CHILDREN)) < 0) {

        report_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                &config->errors, "Invalid value for option %s (%s): %s\n", opt,

                arg, av_err2str(ret));

    } else if ((e = av_dict_get(*dict, option, NULL, 0))) {

        if ((o->type == AV_OPT_TYPE_FLAGS) && arg && (arg[0] == '+' || arg[0] == '-'))

            return av_dict_set(dict, option, arg, AV_DICT_APPEND);

        report_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                &config->errors,

                "Redeclaring value of the option %s, previous value: %s\n",

                opt, e->value);

    } else if (av_dict_set(dict, option, arg, 0) < 0) {

        return AVERROR(ENOMEM);

    }

    return 0;

}
