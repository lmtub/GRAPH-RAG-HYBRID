static int v410_encode_frame(AVCodecContext *avctx, uint8_t *buf,

                             int buf_size, void *data)

{

    AVFrame *pic = data;

    uint8_t *dst = buf;

    uint16_t *y, *u, *v;

    uint32_t val;

    int i, j;

    int output_size = 0;



    if (buf_size < avctx->width * avctx->height * 3) {

        av_log(avctx, AV_LOG_ERROR, "Out buffer is too small.\n");

        return AVERROR(ENOMEM);

    }



    avctx->coded_frame->reference = 0;

    avctx->coded_frame->key_frame = 1;

    avctx->coded_frame->pict_type = FF_I_TYPE;



    y = (uint16_t *)pic->data[0];

    u = (uint16_t *)pic->data[1];

    v = (uint16_t *)pic->data[2];



    for (i = 0; i < avctx->height; i++) {

        for (j = 0; j < avctx->width; j++) {

            val  = u[j] << 2;

            val |= y[j] << 12;

            val |= v[j] << 22;

            AV_WL32(dst, val);

            dst += 4;

            output_size += 4;

        }

        y += pic->linesize[0] >> 1;

        u += pic->linesize[1] >> 1;

        v += pic->linesize[2] >> 1;

    }



    return output_size;

}
