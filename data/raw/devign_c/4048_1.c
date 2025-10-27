static int decode_frame(AVCodecContext * avctx,

                        void *data, int *data_size,

                        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MPADecodeContext *s = avctx->priv_data;

    uint32_t header;

    int out_size;

    OUT_INT *out_samples = data;



    if(buf_size < HEADER_SIZE)

        return -1;



    header = AV_RB32(buf);

    if(ff_mpa_check_header(header) < 0){

        av_log(avctx, AV_LOG_ERROR, "Header missing\n");

        return -1;

    }



    if (ff_mpegaudio_decode_header((MPADecodeHeader *)s, header) == 1) {

        /* free format: prepare to compute frame size */

        s->frame_size = -1;

        return -1;

    }

    /* update codec info */

    avctx->channels = s->nb_channels;

    avctx->bit_rate = s->bit_rate;

    avctx->sub_id = s->layer;



    if(*data_size < 1152*avctx->channels*sizeof(OUT_INT))

        return -1;




    if(s->frame_size<=0 || s->frame_size > buf_size){

        av_log(avctx, AV_LOG_ERROR, "incomplete frame\n");

        return -1;

    }else if(s->frame_size < buf_size){

        av_log(avctx, AV_LOG_ERROR, "incorrect frame size\n");

        buf_size= s->frame_size;

    }



    out_size = mp_decode_frame(s, out_samples, buf, buf_size);

    if(out_size>=0){

        *data_size = out_size;

        avctx->sample_rate = s->sample_rate;

        //FIXME maybe move the other codec info stuff from above here too

    }else

        av_log(avctx, AV_LOG_DEBUG, "Error while decoding MPEG audio frame.\n"); //FIXME return -1 / but also return the number of bytes consumed

    s->frame_size = 0;

    return buf_size;

}