static int encode_frame(AVCodecContext *avctx, uint8_t *buf,

                        int buf_size, void *data)

{

    AVFrame *pic = data;

    int i, j;

    int aligned_width = FFALIGN(avctx->width, 64);

    uint8_t *src_line;

    uint8_t *dst = buf;



    if (buf_size < 4 * aligned_width * avctx->height) {

        av_log(avctx, AV_LOG_ERROR, "output buffer too small\n");

        return AVERROR(ENOMEM);

    }



    avctx->coded_frame->reference = 0;

    avctx->coded_frame->key_frame = 1;

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    src_line = pic->data[0];



    for (i = 0; i < avctx->height; i++) {

        uint16_t *src = (uint16_t *)src_line;

        for (j = 0; j < avctx->width; j++) {

            uint32_t pixel;

            uint16_t r = *src++ >> 6;

            uint16_t g = *src++ >> 6;

            uint16_t b = *src++ >> 4;

            if (avctx->codec_id == CODEC_ID_R210)

                pixel = (r << 20) | (g << 10) | b >> 2;

            else

                pixel = (r << 22) | (g << 12) | b;

            if (avctx->codec_id == CODEC_ID_AVRP)

                bytestream_put_le32(&dst, pixel);

            else

                bytestream_put_be32(&dst, pixel);

        }

        dst += (aligned_width - avctx->width) * 4;

        src_line += pic->linesize[0];

    }



    return 4 * aligned_width * avctx->height;

}
