static int encode_frame(AVCodecContext *avctx,

                        uint8_t *buf, int buf_size,

                        void *data)

{

    int tileno, ret;

    J2kEncoderContext *s = avctx->priv_data;



    // init:

    s->buf = s->buf_start = buf;

    s->buf_end = buf + buf_size;



    s->picture = data;



    s->lambda = s->picture->quality * LAMBDA_SCALE;



    copy_frame(s);

    reinit(s);



    if (s->buf_end - s->buf < 2)

        return -1;

    bytestream_put_be16(&s->buf, J2K_SOC);

    if (ret = put_siz(s))

        return ret;

    if (ret = put_cod(s))

        return ret;

    if (ret = put_qcd(s, 0))

        return ret;



    for (tileno = 0; tileno < s->numXtiles * s->numYtiles; tileno++){

        uint8_t *psotptr;

        if ((psotptr = put_sot(s, tileno)) < 0)

            return psotptr;

        if (s->buf_end - s->buf < 2)

            return -1;

        bytestream_put_be16(&s->buf, J2K_SOD);

        if (ret = encode_tile(s, s->tile + tileno, tileno))

            return ret;

        bytestream_put_be32(&psotptr, s->buf - psotptr + 6);

    }

    if (s->buf_end - s->buf < 2)

        return -1;

    bytestream_put_be16(&s->buf, J2K_EOC);



    av_log(s->avctx, AV_LOG_DEBUG, "end\n");

    return s->buf - s->buf_start;

}
