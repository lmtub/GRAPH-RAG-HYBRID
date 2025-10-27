static int vp5_parse_header(VP56Context *s, const uint8_t *buf, int buf_size)

{

    VP56RangeCoder *c = &s->c;

    int rows, cols;



    ff_vp56_init_range_decoder(&s->c, buf, buf_size);

    s->frames[VP56_FRAME_CURRENT]->key_frame = !vp56_rac_get(c);

    vp56_rac_get(c);

    ff_vp56_init_dequant(s, vp56_rac_gets(c, 6));

    if (s->frames[VP56_FRAME_CURRENT]->key_frame)

    {

        vp56_rac_gets(c, 8);

        if(vp56_rac_gets(c, 5) > 5)

            return AVERROR_INVALIDDATA;

        vp56_rac_gets(c, 2);

        if (vp56_rac_get(c)) {

            av_log(s->avctx, AV_LOG_ERROR, "interlacing not supported\n");

            return AVERROR_PATCHWELCOME;

        }

        rows = vp56_rac_gets(c, 8);  /* number of stored macroblock rows */

        cols = vp56_rac_gets(c, 8);  /* number of stored macroblock cols */

        if (!rows || !cols) {

            av_log(s->avctx, AV_LOG_ERROR, "Invalid size %dx%d\n",

                   cols << 4, rows << 4);

            return AVERROR_INVALIDDATA;

        }

        vp56_rac_gets(c, 8);  /* number of displayed macroblock rows */

        vp56_rac_gets(c, 8);  /* number of displayed macroblock cols */

        vp56_rac_gets(c, 2);

        if (!s->macroblocks || /* first frame */

            16*cols != s->avctx->coded_width ||

            16*rows != s->avctx->coded_height) {

            int ret = ff_set_dimensions(s->avctx, 16 * cols, 16 * rows);

            if (ret < 0)

                return ret;

            return VP56_SIZE_CHANGE;

        }

    } else if (!s->macroblocks)

        return AVERROR_INVALIDDATA;

    return 0;

}
