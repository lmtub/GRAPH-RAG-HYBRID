static int adpcm_decode_init(AVCodecContext * avctx)

{

    ADPCMContext *c = avctx->priv_data;



    if(avctx->channels > 2U){

        return -1;

    }



    c->channel = 0;

    c->status[0].predictor = c->status[1].predictor = 0;

    c->status[0].step_index = c->status[1].step_index = 0;

    c->status[0].step = c->status[1].step = 0;



    switch(avctx->codec->id) {

    case CODEC_ID_ADPCM_CT:

        c->status[0].step = c->status[1].step = 511;

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

    return 0;

}
