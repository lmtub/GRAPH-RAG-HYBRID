static int vp6_parse_header(VP56Context *s, const uint8_t *buf, int buf_size,

                            int *golden_frame)

{

    VP56RangeCoder *c = &s->c;

    int parse_filter_info = 0;

    int coeff_offset = 0;

    int vrt_shift = 0;

    int sub_version;

    int rows, cols;

    int res = 1;

    int separated_coeff = buf[0] & 1;



    s->framep[VP56_FRAME_CURRENT]->key_frame = !(buf[0] & 0x80);

    ff_vp56_init_dequant(s, (buf[0] >> 1) & 0x3F);



    if (s->framep[VP56_FRAME_CURRENT]->key_frame) {

        sub_version = buf[1] >> 3;

        if (sub_version > 8)

            return 0;

        s->filter_header = buf[1] & 0x06;

        if (buf[1] & 1) {

            av_log(s->avctx, AV_LOG_ERROR, "interlacing not supported\n");

            return 0;

        }

        if (separated_coeff || !s->filter_header) {

            coeff_offset = AV_RB16(buf+2) - 2;

            buf += 2;

            buf_size -= 2;

        }



        rows = buf[2];  /* number of stored macroblock rows */

        cols = buf[3];  /* number of stored macroblock cols */

        /* buf[4] is number of displayed macroblock rows */

        /* buf[5] is number of displayed macroblock cols */



        if (!s->macroblocks || /* first frame */

            16*cols != s->avctx->coded_width ||

            16*rows != s->avctx->coded_height) {

            avcodec_set_dimensions(s->avctx, 16*cols, 16*rows);

            if (s->avctx->extradata_size == 1) {

                s->avctx->width  -= s->avctx->extradata[0] >> 4;

                s->avctx->height -= s->avctx->extradata[0] & 0x0F;

            }

            res = 2;

        }



        ff_vp56_init_range_decoder(c, buf+6, buf_size-6);

        vp56_rac_gets(c, 2);



        parse_filter_info = s->filter_header;

        if (sub_version < 8)

            vrt_shift = 5;

        s->sub_version = sub_version;

    } else {

        if (!s->sub_version)

            return 0;



        if (separated_coeff || !s->filter_header) {

            coeff_offset = AV_RB16(buf+1) - 2;

            buf += 2;

            buf_size -= 2;

        }

        ff_vp56_init_range_decoder(c, buf+1, buf_size-1);



        *golden_frame = vp56_rac_get(c);

        if (s->filter_header) {

            s->deblock_filtering = vp56_rac_get(c);

            if (s->deblock_filtering)

                vp56_rac_get(c);

            if (s->sub_version > 7)

                parse_filter_info = vp56_rac_get(c);

        }

    }



    if (parse_filter_info) {

        if (vp56_rac_get(c)) {

            s->filter_mode = 2;

            s->sample_variance_threshold = vp56_rac_gets(c, 5) << vrt_shift;

            s->max_vector_length = 2 << vp56_rac_gets(c, 3);

        } else if (vp56_rac_get(c)) {

            s->filter_mode = 1;

        } else {

            s->filter_mode = 0;

        }

        if (s->sub_version > 7)

            s->filter_selection = vp56_rac_gets(c, 4);

        else

            s->filter_selection = 16;

    }



    s->use_huffman = vp56_rac_get(c);



    s->parse_coeff = vp6_parse_coeff;

    if (coeff_offset) {

        buf      += coeff_offset;

        buf_size -= coeff_offset;

        if (buf_size < 0)

            return 0;

        if (s->use_huffman) {

            s->parse_coeff = vp6_parse_coeff_huffman;

            init_get_bits(&s->gb, buf, buf_size<<3);

        } else {

            ff_vp56_init_range_decoder(&s->cc, buf, buf_size);

            s->ccp = &s->cc;

        }

    } else {

        s->ccp = &s->c;

    }



    return res;

}
