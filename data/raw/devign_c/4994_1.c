static int vp6_parse_header(VP56Context *s, const uint8_t *buf, int buf_size)

{

    VP56RangeCoder *c = &s->c;

    int parse_filter_info = 0;

    int coeff_offset = 0;

    int vrt_shift = 0;

    int sub_version;

    int rows, cols;

    int res = 0;

    int ret;

    int separated_coeff = buf[0] & 1;



    s->frames[VP56_FRAME_CURRENT]->key_frame = !(buf[0] & 0x80);

    ff_vp56_init_dequant(s, (buf[0] >> 1) & 0x3F);



    if (s->frames[VP56_FRAME_CURRENT]->key_frame) {

        sub_version = buf[1] >> 3;

        if (sub_version > 8)

            return AVERROR_INVALIDDATA;

        s->filter_header = buf[1] & 0x06;

        if (buf[1] & 1) {

            avpriv_report_missing_feature(s->avctx, "Interlacing");

            return AVERROR_PATCHWELCOME;

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

        if (!rows || !cols) {

            av_log(s->avctx, AV_LOG_ERROR, "Invalid size %dx%d\n", cols << 4, rows << 4);

            return AVERROR_INVALIDDATA;

        }



        if (!s->macroblocks || /* first frame */

            16*cols != s->avctx->coded_width ||

            16*rows != s->avctx->coded_height) {

            if (s->avctx->extradata_size == 0 &&

                FFALIGN(s->avctx->width,  16) == 16 * cols &&

                FFALIGN(s->avctx->height, 16) == 16 * rows) {

                // We assume this is properly signalled container cropping,

                // in an F4V file. Just set the coded_width/height, don't

                // touch the cropped ones.

                s->avctx->coded_width  = 16 * cols;

                s->avctx->coded_height = 16 * rows;

            } else {

                ret = ff_set_dimensions(s->avctx, 16 * cols, 16 * rows);

                if (ret < 0)

                    return ret;



                if (s->avctx->extradata_size == 1) {

                    s->avctx->width  -= s->avctx->extradata[0] >> 4;

                    s->avctx->height -= s->avctx->extradata[0] & 0x0F;

                }

            }

            res = VP56_SIZE_CHANGE;

        }



        ret = ff_vp56_init_range_decoder(c, buf+6, buf_size-6);

        if (ret < 0)

            return ret;

        vp56_rac_gets(c, 2);



        parse_filter_info = s->filter_header;

        if (sub_version < 8)

            vrt_shift = 5;

        s->sub_version = sub_version;

        s->golden_frame = 0;

    } else {

        if (!s->sub_version || !s->avctx->coded_width || !s->avctx->coded_height)

            return AVERROR_INVALIDDATA;



        if (separated_coeff || !s->filter_header) {

            coeff_offset = AV_RB16(buf+1) - 2;

            buf += 2;

            buf_size -= 2;

        }

        ret = ff_vp56_init_range_decoder(c, buf+1, buf_size-1);

        if (ret < 0)

            return ret;



        s->golden_frame = vp56_rac_get(c);

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

        if (buf_size < 0) {

            if (s->frames[VP56_FRAME_CURRENT]->key_frame)

                ff_set_dimensions(s->avctx, 0, 0);

            return AVERROR_INVALIDDATA;

        }

        if (s->use_huffman) {

            s->parse_coeff = vp6_parse_coeff_huffman;

            init_get_bits(&s->gb, buf, buf_size<<3);

        } else {

            ret = ff_vp56_init_range_decoder(&s->cc, buf, buf_size);

            if (ret < 0)

                return ret;

            s->ccp = &s->cc;

        }

    } else {

        s->ccp = &s->c;

    }



    return res;

}
