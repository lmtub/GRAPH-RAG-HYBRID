static int xan_decode_frame_type0(AVCodecContext *avctx, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    XanContext *s = avctx->priv_data;

    uint8_t *ybuf, *prev_buf, *src = s->scratch_buffer;

    unsigned  chroma_off, corr_off;

    int cur, last, size;

    int i, j;

    int ret;



    corr_off   = AV_RL32(buf + 8);

    chroma_off = AV_RL32(buf + 4);



    if ((ret = xan_decode_chroma(avctx, avpkt)) != 0)

        return ret;



    size = avpkt->size - 4;

    if (corr_off >= avpkt->size) {

        av_log(avctx, AV_LOG_WARNING, "Ignoring invalid correction block position\n");

        corr_off = 0;

    }

    if (corr_off)

        size = corr_off;

    if (chroma_off)

        size = FFMIN(size, chroma_off);

    ret = xan_unpack_luma(buf + 12, size, src, s->buffer_size >> 1);

    if (ret) {

        av_log(avctx, AV_LOG_ERROR, "Luma decoding failed\n");

        return ret;

    }



    ybuf = s->y_buffer;

    last = *src++;

    ybuf[0] = last << 1;

    for (j = 1; j < avctx->width - 1; j += 2) {

        cur = (last + *src++) & 0x1F;

        ybuf[j]   = last + cur;

        ybuf[j+1] = cur << 1;

        last = cur;

    }

    ybuf[j]  = last << 1;

    prev_buf = ybuf;

    ybuf += avctx->width;



    for (i = 1; i < avctx->height; i++) {

        last = ((prev_buf[0] >> 1) + *src++) & 0x1F;

        ybuf[0] = last << 1;

        for (j = 1; j < avctx->width - 1; j += 2) {

            cur = ((prev_buf[j + 1] >> 1) + *src++) & 0x1F;

            ybuf[j]   = last + cur;

            ybuf[j+1] = cur << 1;

            last = cur;

        }

        ybuf[j] = last << 1;

        prev_buf = ybuf;

        ybuf += avctx->width;

    }



    if (corr_off) {

        int corr_end, dec_size;



        corr_end = avpkt->size;

        if (chroma_off > corr_off)

            corr_end = chroma_off;

        dec_size = xan_unpack(s->scratch_buffer, s->buffer_size,

                              avpkt->data + 8 + corr_off,

                              corr_end - corr_off);

        if (dec_size < 0)

            dec_size = 0;

        for (i = 0; i < dec_size; i++)

            s->y_buffer[i*2+1] = (s->y_buffer[i*2+1] + (s->scratch_buffer[i] << 1)) & 0x3F;

    }



    src  = s->y_buffer;

    ybuf = s->pic.data[0];

    for (j = 0; j < avctx->height; j++) {

        for (i = 0; i < avctx->width; i++)

            ybuf[i] = (src[i] << 2) | (src[i] >> 3);

        src  += avctx->width;

        ybuf += s->pic.linesize[0];

    }



    return 0;

}
