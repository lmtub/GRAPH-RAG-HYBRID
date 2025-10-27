static av_cold int libspeex_decode_init(AVCodecContext *avctx)

{

    LibSpeexContext *s = avctx->priv_data;

    const SpeexMode *mode;



    // defaults in the case of a missing header

    if (avctx->sample_rate <= 8000)

        mode = &speex_nb_mode;

    else if (avctx->sample_rate <= 16000)

        mode = &speex_wb_mode;

    else

        mode = &speex_uwb_mode;



    if (avctx->extradata_size >= 80)

        s->header = speex_packet_to_header(avctx->extradata, avctx->extradata_size);



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    if (s->header) {

        avctx->sample_rate = s->header->rate;

        avctx->channels    = s->header->nb_channels;

        s->frame_size      = s->header->frame_size;



        mode = speex_lib_get_mode(s->header->mode);

        if (!mode) {

            av_log(avctx, AV_LOG_ERROR, "Unknown Speex mode %d", s->header->mode);

            return AVERROR_INVALIDDATA;

        }

    } else

        av_log(avctx, AV_LOG_INFO, "Missing Speex header, assuming defaults.\n");



    if (avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "Only stereo and mono are supported.\n");

        return AVERROR(EINVAL);

    }



    speex_bits_init(&s->bits);

    s->dec_state = speex_decoder_init(mode);

    if (!s->dec_state) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing libspeex decoder.\n");

        return -1;

    }



    if (!s->header) {

        speex_decoder_ctl(s->dec_state, SPEEX_GET_FRAME_SIZE, &s->frame_size);

    }



    if (avctx->channels == 2) {

        SpeexCallback callback;

        callback.callback_id = SPEEX_INBAND_STEREO;

        callback.func = speex_std_stereo_request_handler;

        callback.data = &s->stereo;

        s->stereo = (SpeexStereoState)SPEEX_STEREO_STATE_INIT;

        speex_decoder_ctl(s->dec_state, SPEEX_SET_HANDLER, &callback);

    }



    avcodec_get_frame_defaults(&s->frame);

    avctx->coded_frame = &s->frame;



    return 0;

}
