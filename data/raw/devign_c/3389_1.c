static int codec_reinit(AVCodecContext *avctx, int width, int height, int quality) {

    NuvContext *c = avctx->priv_data;

    width = (width + 1) & ~1;

    height = (height + 1) & ~1;

    if (quality >= 0)

        get_quant_quality(c, quality);

    if (width != c->width || height != c->height) {

        if (av_image_check_size(height, width, 0, avctx) < 0)

            return 0;

        avctx->width = c->width = width;

        avctx->height = c->height = height;

        c->decomp_size = c->height * c->width * 3 / 2;

        c->decomp_buf = av_realloc(c->decomp_buf, c->decomp_size + AV_LZO_OUTPUT_PADDING);

        if (!c->decomp_buf) {

            av_log(avctx, AV_LOG_ERROR, "Can't allocate decompression buffer.\n");

            return 0;

        }

        rtjpeg_decode_init(&c->rtj, &c->dsp, c->width, c->height, c->lq, c->cq);

    } else if (quality != c->quality)

        rtjpeg_decode_init(&c->rtj, &c->dsp, c->width, c->height, c->lq, c->cq);

    return 1;

}
