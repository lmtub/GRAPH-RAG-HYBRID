void avcodec_get_context_defaults2(AVCodecContext *s, enum CodecType codec_type){

    int flags=0;

    memset(s, 0, sizeof(AVCodecContext));



    s->av_class= &av_codec_context_class;



    s->codec_type = codec_type;

    if(codec_type == CODEC_TYPE_AUDIO)

        flags= AV_OPT_FLAG_AUDIO_PARAM;

    else if(codec_type == CODEC_TYPE_VIDEO)

        flags= AV_OPT_FLAG_VIDEO_PARAM;

    else if(codec_type == CODEC_TYPE_SUBTITLE)

        flags= AV_OPT_FLAG_SUBTITLE_PARAM;

    av_opt_set_defaults2(s, flags, flags);



    s->rc_eq= av_strdup("tex^qComp");

    s->time_base= (AVRational){0,1};

    s->get_buffer= avcodec_default_get_buffer;

    s->release_buffer= avcodec_default_release_buffer;

    s->get_format= avcodec_default_get_format;

    s->execute= avcodec_default_execute;

    s->sample_aspect_ratio= (AVRational){0,1};

    s->pix_fmt= PIX_FMT_NONE;

    s->sample_fmt= SAMPLE_FMT_S16; // FIXME: set to NONE



    s->palctrl = NULL;

    s->reget_buffer= avcodec_default_reget_buffer;

}
