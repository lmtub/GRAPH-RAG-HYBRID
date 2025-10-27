static int adx_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame_ptr, AVPacket *avpkt)

{

    int buf_size        = avpkt->size;

    ADXContext *c       = avctx->priv_data;

    int16_t *samples;

    const uint8_t *buf  = avpkt->data;

    int num_blocks, ch, ret;



    if (c->eof) {

        *got_frame_ptr = 0;

        return buf_size;

    }



    if (!c->header_parsed && buf_size >= 2 && AV_RB16(buf) == 0x8000) {

        int header_size;

        if ((ret = avpriv_adx_decode_header(avctx, buf, buf_size, &header_size,

                                            c->coeff)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "error parsing ADX header\n");

            return AVERROR_INVALIDDATA;

        }

        c->channels      = avctx->channels;

        c->header_parsed = 1;

        if (buf_size < header_size)

            return AVERROR_INVALIDDATA;

        buf      += header_size;

        buf_size -= header_size;

    }

    if (!c->header_parsed)

        return AVERROR_INVALIDDATA;



    /* calculate number of blocks in the packet */

    num_blocks = buf_size / (BLOCK_SIZE * c->channels);



    /* if the packet is not an even multiple of BLOCK_SIZE, check for an EOF

       packet */

    if (!num_blocks || buf_size % (BLOCK_SIZE * avctx->channels)) {

        if (buf_size >= 4 && (AV_RB16(buf) & 0x8000)) {

            c->eof = 1;

            *got_frame_ptr = 0;

            return avpkt->size;

        }

        return AVERROR_INVALIDDATA;

    }



    /* get output buffer */

    c->frame.nb_samples = num_blocks * BLOCK_SAMPLES;

    if ((ret = avctx->get_buffer(avctx, &c->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    samples = (int16_t *)c->frame.data[0];



    while (num_blocks--) {

        for (ch = 0; ch < c->channels; ch++) {

            if (adx_decode(c, samples + ch, buf, ch)) {

                c->eof = 1;

                buf = avpkt->data + avpkt->size;

                break;

            }

            buf_size -= BLOCK_SIZE;

            buf      += BLOCK_SIZE;

        }

        samples += BLOCK_SAMPLES * c->channels;

    }



    *got_frame_ptr   = 1;

    *(AVFrame *)data = c->frame;



    return buf - avpkt->data;

}
