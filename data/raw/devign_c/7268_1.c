static av_cold int adpcm_decode_init(AVCodecContext * avctx)

{

    ADPCMDecodeContext *c = avctx->priv_data;

    unsigned int max_channels = 2;



    switch(avctx->codec->id) {

    case CODEC_ID_ADPCM_EA_R1:

    case CODEC_ID_ADPCM_EA_R2:

    case CODEC_ID_ADPCM_EA_R3:

    case CODEC_ID_ADPCM_EA_XAS:

        max_channels = 6;

        break;

    }

    if(avctx->channels > max_channels){

        return -1;

    }



    switch(avctx->codec->id) {

    case CODEC_ID_ADPCM_CT:

        c->status[0].step = c->status[1].step = 511;

        break;

    case CODEC_ID_ADPCM_IMA_WAV:

        if (avctx->bits_per_coded_sample != 4) {

            av_log(avctx, AV_LOG_ERROR, "Only 4-bit ADPCM IMA WAV files are supported\n");

            return -1;

        }

        break;

    case CODEC_ID_ADPCM_IMA_WS:

        if (avctx->extradata && avctx->extradata_size == 2 * 4) {

            c->status[0].predictor = AV_RL32(avctx->extradata);

            c->status[1].predictor = AV_RL32(avctx->extradata + 4);

        }

        break;

    default:

        break;

    }

    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    avcodec_get_frame_defaults(&c->frame);

    avctx->coded_frame = &c->frame;



    return 0;

}
