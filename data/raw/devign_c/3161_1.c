static av_cold int amr_wb_encode_init(AVCodecContext *avctx)

{

    AMRWBContext *s = avctx->priv_data;



    if (avctx->sample_rate != 16000) {

        av_log(avctx, AV_LOG_ERROR, "Only 16000Hz sample rate supported\n");

        return AVERROR(ENOSYS);

    }



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "Only mono supported\n");

        return AVERROR(ENOSYS);

    }



    s->mode            = get_wb_bitrate_mode(avctx->bit_rate, avctx);

    s->last_bitrate    = avctx->bit_rate;



    avctx->frame_size  = 320;

    avctx->coded_frame = avcodec_alloc_frame();





    s->state     = E_IF_init();



    return 0;

}