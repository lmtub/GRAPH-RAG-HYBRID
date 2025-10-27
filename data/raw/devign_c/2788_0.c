static av_cold int amr_nb_encode_init(AVCodecContext *avctx)

{

    AMRContext *s = avctx->priv_data;



    if (avctx->sample_rate != 8000) {

        av_log(avctx, AV_LOG_ERROR, "Only 8000Hz sample rate supported\n");

        return AVERROR(ENOSYS);

    }



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "Only mono supported\n");

        return AVERROR(ENOSYS);

    }



    avctx->frame_size  = 160;

    avctx->initial_padding = 50;

    ff_af_queue_init(avctx, &s->afq);



    s->enc_state = Encoder_Interface_init(s->enc_dtx);

    if (!s->enc_state) {

        av_log(avctx, AV_LOG_ERROR, "Encoder_Interface_init error\n");

        av_freep(&avctx->coded_frame);

        return -1;

    }



    s->enc_mode    = get_bitrate_mode(avctx->bit_rate, avctx);

    s->enc_bitrate = avctx->bit_rate;



    return 0;

}
