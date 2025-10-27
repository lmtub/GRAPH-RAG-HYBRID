static int xan_decode_frame_type1(AVCodecContext *avctx)

{

    XanContext *s = avctx->priv_data;

    uint8_t *ybuf, *src = s->scratch_buffer;

    int cur, last;

    int i, j;

    int ret;



    if ((ret = xan_decode_chroma(avctx, bytestream2_get_le32(&s->gb))) != 0)

        return ret;



    bytestream2_seek(&s->gb, 16, SEEK_SET);

    ret = xan_unpack_luma(s, src,

                          s->buffer_size >> 1);

    if (ret) {

        av_log(avctx, AV_LOG_ERROR, "Luma decoding failed\n");

        return ret;

    }



    ybuf = s->y_buffer;

    for (i = 0; i < avctx->height; i++) {

        last = (ybuf[0] + (*src++ << 1)) & 0x3F;

        ybuf[0] = last;

        for (j = 1; j < avctx->width - 1; j += 2) {

            cur = (ybuf[j + 1] + (*src++ << 1)) & 0x3F;

            ybuf[j]   = (last + cur) >> 1;

            ybuf[j+1] = cur;

            last = cur;

        }

        ybuf[j] = last;

        ybuf += avctx->width;

    }



    src = s->y_buffer;

    ybuf = s->pic.data[0];

    for (j = 0; j < avctx->height; j++) {

        for (i = 0; i < avctx->width; i++)

            ybuf[i] = (src[i] << 2) | (src[i] >> 3);

        src  += avctx->width;

        ybuf += s->pic.linesize[0];

    }



    return 0;

}
