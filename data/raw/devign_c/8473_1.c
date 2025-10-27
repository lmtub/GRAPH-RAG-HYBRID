static void vc1_decode_b_mb_intfi(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &s->gb;

    int i, j;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int cbp = 0; /* cbp decoding stuff */

    int mqdiff, mquant; /* MB quantization */

    int ttmb = v->ttfrm; /* MB Transform type */

    int mb_has_coeffs = 0; /* last_flag */

    int val; /* temp value */

    int first_block = 1;

    int dst_idx, off;

    int fwd;

    int dmv_x[2], dmv_y[2], pred_flag[2];

    int bmvtype = BMV_TYPE_BACKWARD;

    int idx_mbmode;

    int av_uninit(interpmvp);



    mquant      = v->pq; /* Lossy initialization */

    s->mb_intra = 0;



    idx_mbmode = get_vlc2(gb, v->mbmode_vlc->table, VC1_IF_MBMODE_VLC_BITS, 2);

    if (idx_mbmode <= 1) { // intra MB

        s->mb_intra = v->is_intra[s->mb_x] = 1;

        s->current_picture.motion_val[1][s->block_index[0]][0] = 0;

        s->current_picture.motion_val[1][s->block_index[0]][1] = 0;

        s->current_picture.mb_type[mb_pos + v->mb_off]         = MB_TYPE_INTRA;

        GET_MQUANT();

        s->current_picture.qscale_table[mb_pos] = mquant;

        /* Set DC scale - y and c use the same (not sure if necessary here) */

        s->y_dc_scale = s->y_dc_scale_table[mquant];

        s->c_dc_scale = s->c_dc_scale_table[mquant];

        v->s.ac_pred  = v->acpred_plane[mb_pos] = get_bits1(gb);

        mb_has_coeffs = idx_mbmode & 1;

        if (mb_has_coeffs)

            cbp = 1 + get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_ICBPCY_VLC_BITS, 2);

        dst_idx = 0;

        for (i = 0; i < 6; i++) {

            s->dc_val[0][s->block_index[i]] = 0;

            dst_idx += i >> 2;

            val = ((cbp >> (5 - i)) & 1);

            v->mb_type[0][s->block_index[i]] = s->mb_intra;

            v->a_avail                       = v->c_avail = 0;

            if (i == 2 || i == 3 || !s->first_slice_line)

                v->a_avail = v->mb_type[0][s->block_index[i] - s->block_wrap[i]];

            if (i == 1 || i == 3 || s->mb_x)

                v->c_avail = v->mb_type[0][s->block_index[i] - 1];



            vc1_decode_intra_block(v, s->block[i], i, val, mquant,

                                   (i & 4) ? v->codingset2 : v->codingset);

            if ((i>3) && (s->flags & CODEC_FLAG_GRAY))

                continue;

            v->vc1dsp.vc1_inv_trans_8x8(s->block[i]);

            if (v->rangeredfrm)

                for (j = 0; j < 64; j++)

                    s->block[i][j] <<= 1;

            off  = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * s->linesize);

            s->dsp.put_signed_pixels_clamped(s->block[i], s->dest[dst_idx] + off, (i & 4) ? s->uvlinesize : s->linesize);

            // TODO: yet to perform loop filter

        }

    } else {

        s->mb_intra = v->is_intra[s->mb_x] = 0;

        s->current_picture.mb_type[mb_pos + v->mb_off] = MB_TYPE_16x16;

        for (i = 0; i < 6; i++) v->mb_type[0][s->block_index[i]] = 0;

        if (v->fmb_is_raw)

            fwd = v->forward_mb_plane[mb_pos] = get_bits1(gb);

        else

            fwd = v->forward_mb_plane[mb_pos];

        if (idx_mbmode <= 5) { // 1-MV

            dmv_x[0]     = dmv_x[1] = dmv_y[0] = dmv_y[1] = 0;

            pred_flag[0] = pred_flag[1] = 0;

            if (fwd)

                bmvtype = BMV_TYPE_FORWARD;

            else {

                bmvtype = decode012(gb);

                switch (bmvtype) {

                case 0:

                    bmvtype = BMV_TYPE_BACKWARD;

                    break;

                case 1:

                    bmvtype = BMV_TYPE_DIRECT;

                    break;

                case 2:

                    bmvtype   = BMV_TYPE_INTERPOLATED;

                    interpmvp = get_bits1(gb);

                }

            }

            v->bmvtype = bmvtype;

            if (bmvtype != BMV_TYPE_DIRECT && idx_mbmode & 1) {

                get_mvdata_interlaced(v, &dmv_x[bmvtype == BMV_TYPE_BACKWARD], &dmv_y[bmvtype == BMV_TYPE_BACKWARD], &pred_flag[bmvtype == BMV_TYPE_BACKWARD]);

            }

            if (bmvtype == BMV_TYPE_INTERPOLATED && interpmvp) {

                get_mvdata_interlaced(v, &dmv_x[1], &dmv_y[1], &pred_flag[1]);

            }

            if (bmvtype == BMV_TYPE_DIRECT) {

                dmv_x[0] = dmv_y[0] = pred_flag[0] = 0;

                dmv_x[1] = dmv_y[1] = pred_flag[0] = 0;

            }

            vc1_pred_b_mv_intfi(v, 0, dmv_x, dmv_y, 1, pred_flag);

            vc1_b_mc(v, dmv_x, dmv_y, (bmvtype == BMV_TYPE_DIRECT), bmvtype);

            mb_has_coeffs = !(idx_mbmode & 2);

        } else { // 4-MV

            if (fwd)

                bmvtype = BMV_TYPE_FORWARD;

            v->bmvtype  = bmvtype;

            v->fourmvbp = get_vlc2(gb, v->fourmvbp_vlc->table, VC1_4MV_BLOCK_PATTERN_VLC_BITS, 1);

            for (i = 0; i < 6; i++) {

                if (i < 4) {

                    dmv_x[0] = dmv_y[0] = pred_flag[0] = 0;

                    dmv_x[1] = dmv_y[1] = pred_flag[1] = 0;

                    val = ((v->fourmvbp >> (3 - i)) & 1);

                    if (val) {

                        get_mvdata_interlaced(v, &dmv_x[bmvtype == BMV_TYPE_BACKWARD],

                                                 &dmv_y[bmvtype == BMV_TYPE_BACKWARD],

                                             &pred_flag[bmvtype == BMV_TYPE_BACKWARD]);

                    }

                    vc1_pred_b_mv_intfi(v, i, dmv_x, dmv_y, 0, pred_flag);

                    vc1_mc_4mv_luma(v, i, bmvtype == BMV_TYPE_BACKWARD, 0);

                } else if (i == 4)

                    vc1_mc_4mv_chroma(v, bmvtype == BMV_TYPE_BACKWARD);

            }

            mb_has_coeffs = idx_mbmode & 1;

        }

        if (mb_has_coeffs)

            cbp = 1 + get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

        if (cbp) {

            GET_MQUANT();

        }

        s->current_picture.qscale_table[mb_pos] = mquant;

        if (!v->ttmbf && cbp) {

            ttmb = get_vlc2(gb, ff_vc1_ttmb_vlc[v->tt_index].table, VC1_TTMB_VLC_BITS, 2);

        }

        dst_idx = 0;

        for (i = 0; i < 6; i++) {

            s->dc_val[0][s->block_index[i]] = 0;

            dst_idx += i >> 2;

            val = ((cbp >> (5 - i)) & 1);

            off = (i & 4) ? 0 : (i & 1) * 8 + (i & 2) * 4 * s->linesize;

            if (val) {

                vc1_decode_p_block(v, s->block[i], i, mquant, ttmb,

                                   first_block, s->dest[dst_idx] + off,

                                   (i & 4) ? s->uvlinesize : s->linesize,

                                   (i & 4) && (s->flags & CODEC_FLAG_GRAY), NULL);

                if (!v->ttmbf && ttmb < 8)

                    ttmb = -1;

                first_block = 0;

            }

        }

    }

}
