static int xan_decode_frame_type0(AVCodecContext *avctx)

{

    XanContext *s = avctx->priv_data;

    uint8_t *ybuf, *prev_buf, *src = s->scratch_buffer;

    unsigned  chroma_off, corr_off;

    int cur, last;

    int i, j;

    int ret;



    chroma_off = bytestream2_get_le32(&s->gb);

    corr_off   = bytestream2_get_le32(&s->gb);



    if ((ret = xan_decode_chroma(avctx, chroma_off)) != 0)

        return ret;



    if (corr_off >= bytestream2_size(&s->gb)) {

        av_log(avctx, AV_LOG_WARNING, "Ignoring invalid correction block position\n");

        corr_off = 0;

    }

    bytestream2_seek(&s->gb, 12, SEEK_SET);

    ret = xan_unpack_luma(s, src, s->buffer_size >> 1);

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

        if(j < avctx->width)

            ybuf[j] = last << 1;

        prev_buf = ybuf;

        ybuf += avctx->width;

    }



    if (corr_off) {

        int dec_size;



        bytestream2_seek(&s->gb, 8 + corr_off, SEEK_SET);

        dec_size = xan_unpack(s, s->scratch_buffer, s->buffer_size);

        if (dec_size < 0)

            dec_size = 0;

        else

            dec_size = FFMIN(dec_size, s->buffer_size/2 - 1);



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
