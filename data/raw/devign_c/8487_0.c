static int libgsm_decode_frame(AVCodecContext *avctx,

                               void *data, int *data_size,

                               AVPacket *avpkt) {

    uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    int out_size = avctx->frame_size * av_get_bytes_per_sample(avctx->sample_fmt);



    if (*data_size < out_size) {

        av_log(avctx, AV_LOG_ERROR, "Output buffer is too small\n");

        return AVERROR(EINVAL);

    }



    if (buf_size < avctx->block_align) {

        av_log(avctx, AV_LOG_ERROR, "Packet is too small\n");

        return AVERROR_INVALIDDATA;

    }



    switch(avctx->codec_id) {

    case CODEC_ID_GSM:

        if(gsm_decode(avctx->priv_data,buf,data)) return -1;

        break;

    case CODEC_ID_GSM_MS:

        if(gsm_decode(avctx->priv_data,buf,data) ||

           gsm_decode(avctx->priv_data,buf+33,((int16_t*)data)+GSM_FRAME_SIZE)) return -1;

    }



    *data_size = out_size;

    return avctx->block_align;

}
