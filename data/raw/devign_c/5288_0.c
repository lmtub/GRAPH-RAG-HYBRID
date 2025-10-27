static int ffserver_parse_config_stream(FFServerConfig *config, const char *cmd, const char **p,

                                        int line_num, FFServerStream **pstream)

{

    char arg[1024], arg2[1024];

    FFServerStream *stream;

    int val;



    av_assert0(pstream);

    stream = *pstream;



    if (!av_strcasecmp(cmd, "<Stream")) {

        char *q;

        FFServerStream *s;

        stream = av_mallocz(sizeof(FFServerStream));

        if (!stream)

            return AVERROR(ENOMEM);

        ffserver_get_arg(stream->filename, sizeof(stream->filename), p);

        q = strrchr(stream->filename, '>');

        if (q)

            *q = '\0';



        for (s = config->first_stream; s; s = s->next) {

            if (!strcmp(stream->filename, s->filename))

                ERROR("Stream '%s' already registered\n", s->filename);

        }



        stream->fmt = ffserver_guess_format(NULL, stream->filename, NULL);

        if (stream->fmt) {

            config->audio_id = stream->fmt->audio_codec;

            config->video_id = stream->fmt->video_codec;

        } else {

            config->audio_id = AV_CODEC_ID_NONE;

            config->video_id = AV_CODEC_ID_NONE;

        }

        *pstream = stream;

        return 0;

    }

    av_assert0(stream);

    if (!av_strcasecmp(cmd, "Feed")) {

        FFServerStream *sfeed;

        ffserver_get_arg(arg, sizeof(arg), p);

        sfeed = config->first_feed;

        while (sfeed) {

            if (!strcmp(sfeed->filename, arg))

                break;

            sfeed = sfeed->next_feed;

        }

        if (!sfeed)

            ERROR("Feed with name '%s' for stream '%s' is not defined\n", arg, stream->filename);

        else

            stream->feed = sfeed;

    } else if (!av_strcasecmp(cmd, "Format")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        if (!strcmp(arg, "status")) {

            stream->stream_type = STREAM_TYPE_STATUS;

            stream->fmt = NULL;

        } else {

            stream->stream_type = STREAM_TYPE_LIVE;

            /* JPEG cannot be used here, so use single frame MJPEG */

            if (!strcmp(arg, "jpeg"))

                strcpy(arg, "mjpeg");

            stream->fmt = ffserver_guess_format(arg, NULL, NULL);

            if (!stream->fmt)

                ERROR("Unknown Format: %s\n", arg);

        }

        if (stream->fmt) {

            config->audio_id = stream->fmt->audio_codec;

            config->video_id = stream->fmt->video_codec;

        }

    } else if (!av_strcasecmp(cmd, "InputFormat")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        stream->ifmt = av_find_input_format(arg);

        if (!stream->ifmt)

            ERROR("Unknown input format: %s\n", arg);

    } else if (!av_strcasecmp(cmd, "FaviconURL")) {

        if (stream->stream_type == STREAM_TYPE_STATUS)

            ffserver_get_arg(stream->feed_filename, sizeof(stream->feed_filename), p);

        else

            ERROR("FaviconURL only permitted for status streams\n");

    } else if (!av_strcasecmp(cmd, "Author")    ||

               !av_strcasecmp(cmd, "Comment")   ||

               !av_strcasecmp(cmd, "Copyright") ||

               !av_strcasecmp(cmd, "Title")) {

        char key[32];

        int i;

        ffserver_get_arg(arg, sizeof(arg), p);

        for (i = 0; i < strlen(cmd); i++)

            key[i] = av_tolower(cmd[i]);

        key[i] = 0;

        WARNING("'%s' option in configuration file is deprecated, "

                "use 'Metadata %s VALUE' instead\n", cmd, key);

        if (av_dict_set(&stream->metadata, key, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "Metadata")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_get_arg(arg2, sizeof(arg2), p);

        if (av_dict_set(&stream->metadata, arg, arg2, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "Preroll")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        stream->prebuffer = atof(arg) * 1000;

    } else if (!av_strcasecmp(cmd, "StartSendOnKey")) {

        stream->send_on_key = 1;

    } else if (!av_strcasecmp(cmd, "AudioCodec")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        config->audio_id = opt_codec(arg, AVMEDIA_TYPE_AUDIO);

        if (config->audio_id == AV_CODEC_ID_NONE)

            ERROR("Unknown AudioCodec: %s\n", arg);

    } else if (!av_strcasecmp(cmd, "VideoCodec")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        config->video_id = opt_codec(arg, AVMEDIA_TYPE_VIDEO);

        if (config->video_id == AV_CODEC_ID_NONE)

            ERROR("Unknown VideoCodec: %s\n", arg);

    } else if (!av_strcasecmp(cmd, "MaxTime")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        stream->max_time = atof(arg) * 1000;

    } else if (!av_strcasecmp(cmd, "AudioBitRate")) {

        float f;

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_float_param(&f, arg, 1000, 0, FLT_MAX, config, line_num, "Invalid %s: %s\n", cmd, arg);

        if (av_dict_set_int(&config->audio_conf, cmd, lrintf(f), 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "AudioChannels")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, 1, 8, config, line_num, "Invalid %s: %s, valid range is 1-8.", cmd, arg);

        if (av_dict_set(&config->audio_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "AudioSampleRate")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, 0, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->audio_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoBitRateRange")) {

        int minrate, maxrate;

        ffserver_get_arg(arg, sizeof(arg), p);

        if (sscanf(arg, "%d-%d", &minrate, &maxrate) == 2) {

            if (av_dict_set_int(&config->video_conf, "VideoBitRateRangeMin", minrate, 0) < 0 ||

                av_dict_set_int(&config->video_conf, "VideoBitRateRangeMax", maxrate, 0) < 0)

                goto nomem;

        } else

            ERROR("Incorrect format for VideoBitRateRange -- should be <min>-<max>: %s\n", arg);

    } else if (!av_strcasecmp(cmd, "Debug")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, INT_MIN, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "Strict")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, INT_MIN, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoBufferSize")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 8*1024, 0, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoBitRateTolerance")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 1000, INT_MIN, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoBitRate")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 1000, 0, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

           goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoSize")) {

        int ret, w, h;

        ffserver_get_arg(arg, sizeof(arg), p);

        ret = av_parse_video_size(&w, &h, arg);

        if (ret < 0)

            ERROR("Invalid video size '%s'\n", arg);

        else if ((w % 16) || (h % 16))

            ERROR("Image size must be a multiple of 16\n");

        if (av_dict_set_int(&config->video_conf, "VideoSizeWidth", w, 0) < 0 ||

            av_dict_set_int(&config->video_conf, "VideoSizeHeight", h, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoFrameRate")) {

        AVRational frame_rate;

        ffserver_get_arg(arg, sizeof(arg), p);

        if (av_parse_video_rate(&frame_rate, arg) < 0) {

            ERROR("Incorrect frame rate: %s\n", arg);

        } else {

            if (av_dict_set_int(&config->video_conf, "VideoFrameRateNum", frame_rate.num, 0) < 0 ||

                av_dict_set_int(&config->video_conf, "VideoFrameRateDen", frame_rate.den, 0) < 0)

                goto nomem;

        }

    } else if (!av_strcasecmp(cmd, "PixelFormat")) {

        enum AVPixelFormat pix_fmt;

        ffserver_get_arg(arg, sizeof(arg), p);

        pix_fmt = av_get_pix_fmt(arg);

        if (pix_fmt == AV_PIX_FMT_NONE)

            ERROR("Unknown pixel format: %s\n", arg);

        if (av_dict_set_int(&config->video_conf, cmd, pix_fmt, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoGopSize")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, INT_MIN, INT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoIntraOnly")) {

        if (av_dict_set(&config->video_conf, cmd, "1", 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoHighQuality")) {

        if (av_dict_set(&config->video_conf, cmd, "", 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "Video4MotionVector")) {

        if (av_dict_set(&config->video_conf, cmd, "", 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "AVOptionVideo") ||

               !av_strcasecmp(cmd, "AVOptionAudio")) {

        AVDictionary **dict;

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_get_arg(arg2, sizeof(arg2), p);

        if (!av_strcasecmp(cmd, "AVOptionVideo"))

            dict = &config->video_opts;

        else

            dict = &config->audio_opts;

        if (av_dict_set(dict, arg, arg2, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "AVPresetVideo") ||

               !av_strcasecmp(cmd, "AVPresetAudio")) {

        char **preset = NULL;

        ffserver_get_arg(arg, sizeof(arg), p);

        if (!av_strcasecmp(cmd, "AVPresetVideo")) {

            preset = &config->video_preset;

            ffserver_opt_preset(arg, NULL, 0, NULL, &config->video_id);

        } else {

            preset = &config->audio_preset;

            ffserver_opt_preset(arg, NULL, 0, &config->audio_id, NULL);

        }

        *preset = av_strdup(arg);

        if (!preset)

            return AVERROR(ENOMEM);

    } else if (!av_strcasecmp(cmd, "VideoTag")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        if (strlen(arg) == 4) {

            if (av_dict_set(&config->video_conf, "VideoTag", "arg", 0) < 0)

                goto nomem;

        }

    } else if (!av_strcasecmp(cmd, "BitExact")) {

        if (av_dict_set(&config->video_conf, cmd, "", 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "DctFastint")) {

        if (av_dict_set(&config->video_conf, cmd, "", 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "IdctSimple")) {

        if (av_dict_set(&config->video_conf, cmd, "", 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "Qscale")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoQDiff")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, 1, 31, config, line_num, "%s out of range\n", cmd);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoQMax")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, 1, 31, config, line_num, "%s out of range\n", cmd);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "VideoQMin")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(NULL, arg, 0, 1, 31, config, line_num, "%s out of range\n", cmd);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "LumiMask")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_float_param(NULL, arg, 0, -FLT_MAX, FLT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "DarkMask")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_float_param(NULL, arg, 0, -FLT_MAX, FLT_MAX, config, line_num, "Invalid %s: %s", cmd, arg);

        if (av_dict_set(&config->video_conf, cmd, arg, 0) < 0)

            goto nomem;

    } else if (!av_strcasecmp(cmd, "NoVideo")) {

        config->video_id = AV_CODEC_ID_NONE;

    } else if (!av_strcasecmp(cmd, "NoAudio")) {

        config->audio_id = AV_CODEC_ID_NONE;

    } else if (!av_strcasecmp(cmd, "ACL")) {

        ffserver_parse_acl_row(stream, NULL, NULL, *p, config->filename, line_num);

    } else if (!av_strcasecmp(cmd, "DynamicACL")) {

        ffserver_get_arg(stream->dynamic_acl, sizeof(stream->dynamic_acl), p);

    } else if (!av_strcasecmp(cmd, "RTSPOption")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        av_freep(&stream->rtsp_option);

        stream->rtsp_option = av_strdup(arg);

    } else if (!av_strcasecmp(cmd, "MulticastAddress")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        if (resolve_host(&stream->multicast_ip, arg) != 0)

            ERROR("Invalid host/IP address: %s\n", arg);

        stream->is_multicast = 1;

        stream->loop = 1; /* default is looping */

    } else if (!av_strcasecmp(cmd, "MulticastPort")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(&val, arg, 0, 1, 65535, config, line_num, "Invalid MulticastPort: %s\n", arg);

        stream->multicast_port = val;

    } else if (!av_strcasecmp(cmd, "MulticastTTL")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        ffserver_set_int_param(&val, arg, 0, INT_MIN, INT_MAX, config, line_num, "Invalid MulticastTTL: %s\n", arg);

        stream->multicast_ttl = val;

    } else if (!av_strcasecmp(cmd, "NoLoop")) {

        stream->loop = 0;

    } else if (!av_strcasecmp(cmd, "</Stream>")) {

        if (stream->feed && stream->fmt && strcmp(stream->fmt->name, "ffm") != 0) {

            if (config->audio_id != AV_CODEC_ID_NONE) {

                AVCodecContext *audio_enc = avcodec_alloc_context3(avcodec_find_encoder(config->audio_id));

                if (config->audio_preset &&

                    ffserver_opt_preset(arg, audio_enc, AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_ENCODING_PARAM,

                                        NULL, NULL) < 0)

                    ERROR("Could not apply preset '%s'\n", arg);

                if (ffserver_apply_stream_config(audio_enc, config->audio_conf, &config->audio_opts) < 0)

                    config->errors++;

                add_codec(stream, audio_enc);

            }

            if (config->video_id != AV_CODEC_ID_NONE) {

                AVCodecContext *video_enc = avcodec_alloc_context3(avcodec_find_encoder(config->video_id));

                if (config->video_preset &&

                    ffserver_opt_preset(arg, video_enc, AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_ENCODING_PARAM,

                                        NULL, NULL) < 0)

                    ERROR("Could not apply preset '%s'\n", arg);

                if (ffserver_apply_stream_config(video_enc, config->video_conf, &config->video_opts) < 0)

                    config->errors++;

                add_codec(stream, video_enc);

            }

        }

        av_dict_free(&config->video_opts);

        av_dict_free(&config->video_conf);

        av_dict_free(&config->audio_opts);

        av_dict_free(&config->audio_conf);

        av_freep(&config->video_preset);

        av_freep(&config->audio_preset);

        *pstream = NULL;

    } else if (!av_strcasecmp(cmd, "File") || !av_strcasecmp(cmd, "ReadOnlyFile")) {

        ffserver_get_arg(stream->feed_filename, sizeof(stream->feed_filename), p);

    } else {

        ERROR("Invalid entry '%s' inside <Stream></Stream>\n", cmd);

    }

    return 0;

  nomem:

    av_log(NULL, AV_LOG_ERROR, "Out of memory. Aborting.\n");

    av_dict_free(&config->video_opts);

    av_dict_free(&config->video_conf);

    av_dict_free(&config->audio_opts);

    av_dict_free(&config->audio_conf);

    av_freep(&config->video_preset);

    av_freep(&config->audio_preset);

    return AVERROR(ENOMEM);

}
