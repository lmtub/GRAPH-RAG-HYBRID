static int tiff_unpack_fax(TiffContext *s, uint8_t *dst, int stride,

                           const uint8_t *src, int size, int width, int lines)

{

    int i, ret = 0;

    int line;

    uint8_t *src2 = av_malloc((unsigned)size +

                              AV_INPUT_BUFFER_PADDING_SIZE);



    if (!src2) {

        av_log(s->avctx, AV_LOG_ERROR,

               "Error allocating temporary buffer\n");

        return AVERROR(ENOMEM);

    }



    if (!s->fill_order) {

        memcpy(src2, src, size);

    } else {

        for (i = 0; i < size; i++)

            src2[i] = ff_reverse[src[i]];

    }

    memset(src2 + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    ret = ff_ccitt_unpack(s->avctx, src2, size, dst, lines, stride,

                          s->compr, s->fax_opts);

    if (s->bpp < 8 && s->avctx->pix_fmt == AV_PIX_FMT_PAL8)

        for (line = 0; line < lines; line++) {

            horizontal_fill(s->bpp, dst, 1, dst, 0, width, 0);

            dst += stride;

        }

    av_free(src2);

    return ret;

}
