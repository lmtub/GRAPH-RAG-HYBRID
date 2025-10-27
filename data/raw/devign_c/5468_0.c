static av_cold int eightsvx_decode_init(AVCodecContext *avctx)

{

    EightSvxContext *esc = avctx->priv_data;



    if (avctx->channels < 1 || avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "8SVX does not support more than 2 channels\n");

        return AVERROR_INVALIDDATA;

    }



    switch (avctx->codec->id) {

    case AV_CODEC_ID_8SVX_FIB: esc->table = fibonacci;    break;

    case AV_CODEC_ID_8SVX_EXP: esc->table = exponential;  break;

    case AV_CODEC_ID_PCM_S8_PLANAR:

    case AV_CODEC_ID_8SVX_RAW: esc->table = NULL;         break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Invalid codec id %d.\n", avctx->codec->id);

        return AVERROR_INVALIDDATA;

    }

    avctx->sample_fmt = AV_SAMPLE_FMT_U8P;



    avcodec_get_frame_defaults(&esc->frame);

    avctx->coded_frame = &esc->frame;



    return 0;

}
