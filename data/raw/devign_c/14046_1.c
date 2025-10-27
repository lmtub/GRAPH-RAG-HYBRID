int msmpeg4_decode_mb(MpegEncContext *s, 

                      DCTELEM block[6][64])

{

    int cbp, code, i;

    int pred, val;

    UINT8 *coded_val;



    /* special slice handling */

    if (s->mb_x == 0) {

        if ((s->mb_y % s->slice_height) == 0) {

            int wrap;

            /* reset DC pred (set previous line to 1024) */

            wrap = 2 * s->mb_width + 2;

            memsetw(&s->dc_val[0][(1) + (2 * s->mb_y) * wrap], 

                    1024, 2 * s->mb_width);

            wrap = s->mb_width + 2;

            memsetw(&s->dc_val[1][(1) + (s->mb_y) * wrap], 

                    1024, s->mb_width);

            memsetw(&s->dc_val[2][(1) + (s->mb_y) * wrap], 

                    1024, s->mb_width);



            s->first_slice_line = 1;

        } else {

            s->first_slice_line = 0; 

        }

    }



    if (s->pict_type == P_TYPE) {

        set_stat(ST_INTER_MB);

        if (s->use_skip_mb_code) {

            if (get_bits1(&s->gb)) {

                /* skip mb */

                s->mb_intra = 0;

                for(i=0;i<6;i++)

                    s->block_last_index[i] = -1;

                s->mv_dir = MV_DIR_FORWARD;

                s->mv_type = MV_TYPE_16X16;

                s->mv[0][0][0] = 0;

                s->mv[0][0][1] = 0;

                s->mb_skiped = 1;

                return 0;

            }

        }

        

        code = get_vlc(&s->gb, &mb_non_intra_vlc);

        if (code < 0)

            return -1;

        if (code & 0x40)

            s->mb_intra = 0;

        else

            s->mb_intra = 1;

            

        cbp = code & 0x3f;

    } else {

        set_stat(ST_INTRA_MB);

        s->mb_intra = 1;

        code = get_vlc(&s->gb, &mb_intra_vlc);

        if (code < 0)

            return -1;

        /* predict coded block pattern */

        cbp = 0;

        for(i=0;i<6;i++) {

            val = ((code >> (5 - i)) & 1);

            if (i < 4) {

                pred = coded_block_pred(s, i, &coded_val);

                val = val ^ pred;

                *coded_val = val;

            }

            cbp |= val << (5 - i);

        }

    }



    if (!s->mb_intra) {

        int mx, my;

        set_stat(ST_MV);

        h263_pred_motion(s, 0, &mx, &my);

        if (msmpeg4_decode_motion(s, &mx, &my) < 0)

            return -1;

        s->mv_dir = MV_DIR_FORWARD;

        s->mv_type = MV_TYPE_16X16;

        s->mv[0][0][0] = mx;

        s->mv[0][0][1] = my;

    } else {

        set_stat(ST_INTRA_MB);

        s->ac_pred = get_bits1(&s->gb);

    }



    for (i = 0; i < 6; i++) {

        if (msmpeg4_decode_block(s, block[i], i, (cbp >> (5 - i)) & 1) < 0)

            return -1;

    }

    return 0;

}
