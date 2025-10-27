static int caca_write_header(AVFormatContext *s)

{

    CACAContext *c = s->priv_data;

    AVStream *st = s->streams[0];

    AVCodecContext *encctx = st->codec;

    int ret, bpp;



    c->ctx = s;

    if (c->list_drivers) {

        list_drivers(c);

        return AVERROR_EXIT;

    }

    if (c->list_dither) {

        if (!strcmp(c->list_dither, "colors")) {

            list_dither_color(c);

        } else if (!strcmp(c->list_dither, "charsets")) {

            list_dither_charset(c);

        } else if (!strcmp(c->list_dither, "algorithms")) {

            list_dither_algorithm(c);

        } else if (!strcmp(c->list_dither, "antialiases")) {

            list_dither_antialias(c);

        } else {

            av_log(s, AV_LOG_ERROR,

                   "Invalid argument '%s', for 'list_dither' option\n"

                   "Argument must be one of 'algorithms, 'antialiases', 'charsets', 'colors'\n",

                   c->list_dither);

            return AVERROR(EINVAL);

        }

        return AVERROR_EXIT;

    }



    if (   s->nb_streams > 1

        || encctx->codec_type != AVMEDIA_TYPE_VIDEO

        || encctx->codec_id   != CODEC_ID_RAWVIDEO) {

        av_log(s, AV_LOG_ERROR, "Only supports one rawvideo stream\n");

        return AVERROR(EINVAL);

    }



    if (encctx->pix_fmt != PIX_FMT_RGB24) {

        av_log(s, AV_LOG_ERROR,

               "Unsupported pixel format '%s', choose rgb24\n",

               av_get_pix_fmt_name(encctx->pix_fmt));

        return AVERROR(EINVAL);

    }



    c->canvas = caca_create_canvas(c->window_width, c->window_height);

    if (!c->canvas) {

        av_log(s, AV_LOG_ERROR, "Failed to create canvas\n");

        return AVERROR(errno);

    }



    c->display = caca_create_display_with_driver(c->canvas, c->driver);

    if (!c->display) {

        av_log(s, AV_LOG_ERROR, "Failed to create display\n");

        list_drivers(c);

        caca_free_canvas(c->canvas);

        return AVERROR(errno);

    }



    if (!c->window_width || !c->window_height) {

        c->window_width  = caca_get_canvas_width(c->canvas);

        c->window_height = caca_get_canvas_height(c->canvas);

    }



    bpp = av_get_bits_per_pixel(&av_pix_fmt_descriptors[encctx->pix_fmt]);

    c->dither = caca_create_dither(bpp, encctx->width, encctx->height,

                                   bpp / 8 * encctx->width,

                                   0x0000ff, 0x00ff00, 0xff0000, 0);

    if (!c->dither) {

        av_log(s, AV_LOG_ERROR, "Failed to create dither\n");

        ret =  AVERROR(errno);

        goto fail;

    }



#define CHECK_DITHER_OPT(opt)                                           \

    if (caca_set_dither_##opt(c->dither, c->opt) < 0)  {                \

        ret = AVERROR(errno);                                           \

        av_log(s, AV_LOG_ERROR, "Failed to set value '%s' for option '%s'\n", \

               c->opt, #opt);                                           \

        goto fail;                                                      \

    }

    CHECK_DITHER_OPT(algorithm);

    CHECK_DITHER_OPT(antialias);

    CHECK_DITHER_OPT(charset);

    CHECK_DITHER_OPT(color);



    if (!c->window_title)

        c->window_title = av_strdup(s->filename);

    caca_set_display_title(c->display, c->window_title);

    caca_set_display_time(c->display, av_rescale_q(1, st->codec->time_base, AV_TIME_BASE_Q));



    return 0;



fail:

    caca_write_trailer(s);

    return ret;

}
