static av_cold int adpcm_decode_init(AVCodecContext * avctx)

{

    ADPCMDecodeContext *c = avctx->priv_data;

    unsigned int min_channels = 1;

    unsigned int max_channels = 2;



    switch(avctx->codec->id) {

    case AV_CODEC_ID_ADPCM_DTK:

    case AV_CODEC_ID_ADPCM_EA:

        min_channels = 2;

        break;

    case AV_CODEC_ID_ADPCM_AFC:

    case AV_CODEC_ID_ADPCM_EA_R1:

    case AV_CODEC_ID_ADPCM_EA_R2:

    case AV_CODEC_ID_ADPCM_EA_R3:

    case AV_CODEC_ID_ADPCM_EA_XAS:

        max_channels = 6;

        break;

    case AV_CODEC_ID_ADPCM_THP:

    case AV_CODEC_ID_ADPCM_THP_LE:

        max_channels = 10;

        break;

    }

    if (avctx->channels < min_channels || avctx->channels > max_channels) {

        av_log(avctx, AV_LOG_ERROR, "Invalid number of channels\n");

        return AVERROR(EINVAL);

    }



    switch(avctx->codec->id) {

    case AV_CODEC_ID_ADPCM_CT:

        c->status[0].step = c->status[1].step = 511;

        break;

    case AV_CODEC_ID_ADPCM_IMA_WAV:

        if (avctx->bits_per_coded_sample < 2 || avctx->bits_per_coded_sample > 5)

            return AVERROR_INVALIDDATA;

        break;

    case AV_CODEC_ID_ADPCM_IMA_APC:

        if (avctx->extradata && avctx->extradata_size >= 8) {

            c->status[0].predictor = AV_RL32(avctx->extradata);

            c->status[1].predictor = AV_RL32(avctx->extradata + 4);

        }

        break;

    case AV_CODEC_ID_ADPCM_IMA_WS:

        if (avctx->extradata && avctx->extradata_size >= 2)

            c->vqa_version = AV_RL16(avctx->extradata);

        break;

    default:

        break;

    }



    switch(avctx->codec->id) {

        case AV_CODEC_ID_ADPCM_IMA_QT:

        case AV_CODEC_ID_ADPCM_IMA_WAV:

        case AV_CODEC_ID_ADPCM_4XM:

        case AV_CODEC_ID_ADPCM_XA:

        case AV_CODEC_ID_ADPCM_EA_R1:

        case AV_CODEC_ID_ADPCM_EA_R2:

        case AV_CODEC_ID_ADPCM_EA_R3:

        case AV_CODEC_ID_ADPCM_EA_XAS:

        case AV_CODEC_ID_ADPCM_THP:

        case AV_CODEC_ID_ADPCM_THP_LE:

        case AV_CODEC_ID_ADPCM_AFC:

        case AV_CODEC_ID_ADPCM_DTK:

            avctx->sample_fmt = AV_SAMPLE_FMT_S16P;

            break;

        case AV_CODEC_ID_ADPCM_IMA_WS:

            avctx->sample_fmt = c->vqa_version == 3 ? AV_SAMPLE_FMT_S16P :

                                                      AV_SAMPLE_FMT_S16;

            break;

        default:

            avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    }



    return 0;

}
