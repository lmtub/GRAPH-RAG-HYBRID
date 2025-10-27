static int read_ffserver_streams(OptionsContext *o, AVFormatContext *s, const char *filename)

{

    int i, err;

    AVFormatContext *ic = avformat_alloc_context();





    ic->interrupt_callback = int_cb;

    err = avformat_open_input(&ic, filename, NULL, NULL);

    if (err < 0)

        return err;

    /* copy stream format */

    for(i=0;i<ic->nb_streams;i++) {

        AVStream *st;

        OutputStream *ost;

        AVCodec *codec;

        const char *enc_config;



        codec = avcodec_find_encoder(ic->streams[i]->codecpar->codec_id);

        if (!codec) {

            av_log(s, AV_LOG_ERROR, "no encoder found for codec id %i\n", ic->streams[i]->codecpar->codec_id);

            return AVERROR(EINVAL);

        }

        if (codec->type == AVMEDIA_TYPE_AUDIO)

            opt_audio_codec(o, "c:a", codec->name);

        else if (codec->type == AVMEDIA_TYPE_VIDEO)

            opt_video_codec(o, "c:v", codec->name);

        ost   = new_output_stream(o, s, codec->type, -1);

        st    = ost->st;



        avcodec_get_context_defaults3(st->codec, codec);

        enc_config = av_stream_get_recommended_encoder_configuration(ic->streams[i]);

        if (enc_config) {

            AVDictionary *opts = NULL;

            av_dict_parse_string(&opts, enc_config, "=", ",", 0);

            av_opt_set_dict2(st->codec, &opts, AV_OPT_SEARCH_CHILDREN);

            av_dict_free(&opts);

        }



        if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && !ost->stream_copy)

            choose_sample_fmt(st, codec);

        else if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !ost->stream_copy)

            choose_pixel_fmt(st, st->codec, codec, st->codecpar->format);

        avcodec_copy_context(ost->enc_ctx, st->codec);

        if (enc_config)

            av_dict_parse_string(&ost->encoder_opts, enc_config, "=", ",", 0);

    }



    avformat_close_input(&ic);

    return err;

}