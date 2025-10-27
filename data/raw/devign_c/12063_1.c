static int vp5_parse_header(VP56Context *s, const uint8_t *buf, int buf_size,

                            int *golden_frame)

{

    VP56RangeCoder *c = &s->c;

    int rows, cols;



    ff_vp56_init_range_decoder(&s->c, buf, buf_size);

    s->framep[VP56_FRAME_CURRENT]->key_frame = !vp56_rac_get(c);

    vp56_rac_get(c);

    ff_vp56_init_dequant(s, vp56_rac_gets(c, 6));

    if (s->framep[VP56_FRAME_CURRENT]->key_frame)

    {

        vp56_rac_gets(c, 8);

        if(vp56_rac_gets(c, 5) > 5)


        vp56_rac_gets(c, 2);

        if (vp56_rac_get(c)) {

            av_log(s->avctx, AV_LOG_ERROR, "interlacing not supported\n");



        rows = vp56_rac_gets(c, 8);  /* number of stored macroblock rows */

        cols = vp56_rac_gets(c, 8);  /* number of stored macroblock cols */






        vp56_rac_gets(c, 8);  /* number of displayed macroblock rows */

        vp56_rac_gets(c, 8);  /* number of displayed macroblock cols */

        vp56_rac_gets(c, 2);

        if (!s->macroblocks || /* first frame */

            16*cols != s->avctx->coded_width ||

            16*rows != s->avctx->coded_height) {

            avcodec_set_dimensions(s->avctx, 16*cols, 16*rows);

            return 2;


    } else if (!s->macroblocks)


    return 1;
