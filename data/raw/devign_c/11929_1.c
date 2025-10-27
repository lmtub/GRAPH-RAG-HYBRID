static int vp8_lossless_decode_frame(AVCodecContext *avctx, AVFrame *p,

                                     int *got_frame, uint8_t *data_start,

                                     unsigned int data_size, int is_alpha_chunk)

{

    WebPContext *s = avctx->priv_data;

    int w, h, ret, i;



    if (!is_alpha_chunk) {

        s->lossless = 1;

        avctx->pix_fmt = AV_PIX_FMT_ARGB;

    }



    ret = init_get_bits8(&s->gb, data_start, data_size);

    if (ret < 0)

        return ret;



    if (!is_alpha_chunk) {

        if (get_bits(&s->gb, 8) != 0x2F) {

            av_log(avctx, AV_LOG_ERROR, "Invalid WebP Lossless signature\n");

            return AVERROR_INVALIDDATA;

        }



        w = get_bits(&s->gb, 14) + 1;

        h = get_bits(&s->gb, 14) + 1;

        if (s->width && s->width != w) {

            av_log(avctx, AV_LOG_WARNING, "Width mismatch. %d != %d\n",

                   s->width, w);

        }

        s->width = w;

        if (s->height && s->height != h) {

            av_log(avctx, AV_LOG_WARNING, "Height mismatch. %d != %d\n",

                   s->width, w);

        }

        s->height = h;



        ret = ff_set_dimensions(avctx, s->width, s->height);

        if (ret < 0)

            return ret;



        s->has_alpha = get_bits1(&s->gb);



        if (get_bits(&s->gb, 3) != 0x0) {

            av_log(avctx, AV_LOG_ERROR, "Invalid WebP Lossless version\n");

            return AVERROR_INVALIDDATA;

        }

    } else {

        if (!s->width || !s->height)

            return AVERROR_BUG;

        w = s->width;

        h = s->height;

    }



    /* parse transformations */

    s->nb_transforms = 0;

    s->reduced_width = 0;

    while (get_bits1(&s->gb)) {

        enum TransformType transform = get_bits(&s->gb, 2);

        s->transforms[s->nb_transforms++] = transform;

        switch (transform) {

        case PREDICTOR_TRANSFORM:

            ret = parse_transform_predictor(s);

            break;

        case COLOR_TRANSFORM:

            ret = parse_transform_color(s);

            break;

        case COLOR_INDEXING_TRANSFORM:

            ret = parse_transform_color_indexing(s);

            break;

        }

        if (ret < 0)

            goto free_and_return;

    }



    /* decode primary image */

    s->image[IMAGE_ROLE_ARGB].frame = p;

    if (is_alpha_chunk)

        s->image[IMAGE_ROLE_ARGB].is_alpha_primary = 1;

    ret = decode_entropy_coded_image(s, IMAGE_ROLE_ARGB, w, h);

    if (ret < 0)

        goto free_and_return;



    /* apply transformations */

    for (i = s->nb_transforms - 1; i >= 0; i--) {

        switch (s->transforms[i]) {

        case PREDICTOR_TRANSFORM:

            ret = apply_predictor_transform(s);

            break;

        case COLOR_TRANSFORM:

            ret = apply_color_transform(s);

            break;

        case SUBTRACT_GREEN:

            ret = apply_subtract_green_transform(s);

            break;

        case COLOR_INDEXING_TRANSFORM:

            ret = apply_color_indexing_transform(s);

            break;

        }

        if (ret < 0)

            goto free_and_return;

    }



    *got_frame   = 1;

    p->pict_type = AV_PICTURE_TYPE_I;

    p->key_frame = 1;

    ret          = data_size;



free_and_return:

    for (i = 0; i < IMAGE_ROLE_NB; i++)

        image_ctx_free(&s->image[i]);



    return ret;

}
