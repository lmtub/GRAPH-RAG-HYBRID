static int vc1_decode_b_mb_intfr(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &s->gb;

    int i, j;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int cbp = 0; /* cbp decoding stuff */

    int mqdiff, mquant; /* MB quantization */

    int ttmb = v->ttfrm; /* MB Transform type */

    int mvsw = 0; /* motion vector switch */

    int mb_has_coeffs = 1; /* last_flag */

    int dmv_x, dmv_y; /* Differential MV components */

    int val; /* temp value */

    int first_block = 1;

    int dst_idx, off;

    int skipped, direct, twomv = 0;

    int block_cbp = 0, pat, block_tt = 0;

    int idx_mbmode = 0, mvbp;

    int stride_y, fieldtx;

    int bmvtype = BMV_TYPE_BACKWARD;

    int dir, dir2;



    mquant = v->pq; /* Lossy initialization */

    s->mb_intra = 0;

    if (v->skip_is_raw)

        skipped = get_bits1(gb);

    else

        skipped = v->s.mbskip_table[mb_pos];



    if (!skipped) {

        idx_mbmode = get_vlc2(gb, v->mbmode_vlc->table, VC1_INTFR_NON4MV_MBMODE_VLC_BITS, 2);

        if (ff_vc1_mbmode_intfrp[0][idx_mbmode][0] == MV_PMODE_INTFR_2MV_FIELD) {

            twomv = 1;

            v->blk_mv_type[s->block_index[0]] = 1;

            v->blk_mv_type[s->block_index[1]] = 1;

            v->blk_mv_type[s->block_index[2]] = 1;

            v->blk_mv_type[s->block_index[3]] = 1;

        } else {

            v->blk_mv_type[s->block_index[0]] = 0;

            v->blk_mv_type[s->block_index[1]] = 0;

            v->blk_mv_type[s->block_index[2]] = 0;

            v->blk_mv_type[s->block_index[3]] = 0;

        }

    }



    if (v->dmb_is_raw)

        direct = get_bits1(gb);

    else

        direct = v->direct_mb_plane[mb_pos];



    if (direct) {

        s->mv[0][0][0] = s->current_picture.motion_val[0][s->block_index[0]][0] = scale_mv(s->next_picture.motion_val[1][s->block_index[0]][0], v->bfraction, 0, s->quarter_sample);

        s->mv[0][0][1] = s->current_picture.motion_val[0][s->block_index[0]][1] = scale_mv(s->next_picture.motion_val[1][s->block_index[0]][1], v->bfraction, 0, s->quarter_sample);

        s->mv[1][0][0] = s->current_picture.motion_val[1][s->block_index[0]][0] = scale_mv(s->next_picture.motion_val[1][s->block_index[0]][0], v->bfraction, 1, s->quarter_sample);

        s->mv[1][0][1] = s->current_picture.motion_val[1][s->block_index[0]][1] = scale_mv(s->next_picture.motion_val[1][s->block_index[0]][1], v->bfraction, 1, s->quarter_sample);



        if (twomv) {

            s->mv[0][2][0] = s->current_picture.motion_val[0][s->block_index[2]][0] = scale_mv(s->next_picture.motion_val[1][s->block_index[2]][0], v->bfraction, 0, s->quarter_sample);

            s->mv[0][2][1] = s->current_picture.motion_val[0][s->block_index[2]][1] = scale_mv(s->next_picture.motion_val[1][s->block_index[2]][1], v->bfraction, 0, s->quarter_sample);

            s->mv[1][2][0] = s->current_picture.motion_val[1][s->block_index[2]][0] = scale_mv(s->next_picture.motion_val[1][s->block_index[2]][0], v->bfraction, 1, s->quarter_sample);

            s->mv[1][2][1] = s->current_picture.motion_val[1][s->block_index[2]][1] = scale_mv(s->next_picture.motion_val[1][s->block_index[2]][1], v->bfraction, 1, s->quarter_sample);



            for (i = 1; i < 4; i += 2) {

                s->mv[0][i][0] = s->current_picture.motion_val[0][s->block_index[i]][0] = s->mv[0][i-1][0];

                s->mv[0][i][1] = s->current_picture.motion_val[0][s->block_index[i]][1] = s->mv[0][i-1][1];

                s->mv[1][i][0] = s->current_picture.motion_val[1][s->block_index[i]][0] = s->mv[1][i-1][0];

                s->mv[1][i][1] = s->current_picture.motion_val[1][s->block_index[i]][1] = s->mv[1][i-1][1];

            }

        } else {

            for (i = 1; i < 4; i++) {

                s->mv[0][i][0] = s->current_picture.motion_val[0][s->block_index[i]][0] = s->mv[0][0][0];

                s->mv[0][i][1] = s->current_picture.motion_val[0][s->block_index[i]][1] = s->mv[0][0][1];

                s->mv[1][i][0] = s->current_picture.motion_val[1][s->block_index[i]][0] = s->mv[1][0][0];

                s->mv[1][i][1] = s->current_picture.motion_val[1][s->block_index[i]][1] = s->mv[1][0][1];

            }

        }

    }



    if (ff_vc1_mbmode_intfrp[0][idx_mbmode][0] == MV_PMODE_INTFR_INTRA) { // intra MB

        for (i = 0; i < 4; i++) {

            s->mv[0][i][0] = s->current_picture.motion_val[0][s->block_index[i]][0] = 0;

            s->mv[0][i][1] = s->current_picture.motion_val[0][s->block_index[i]][1] = 0;

            s->mv[1][i][0] = s->current_picture.motion_val[1][s->block_index[i]][0] = 0;

            s->mv[1][i][1] = s->current_picture.motion_val[1][s->block_index[i]][1] = 0;

        }

        s->current_picture.mb_type[mb_pos] = MB_TYPE_INTRA;

        s->mb_intra = v->is_intra[s->mb_x] = 1;

        for (i = 0; i < 6; i++)

            v->mb_type[0][s->block_index[i]] = 1;

        fieldtx = v->fieldtx_plane[mb_pos] = get_bits1(gb);

        mb_has_coeffs = get_bits1(gb);

        if (mb_has_coeffs)

            cbp = 1 + get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

        v->s.ac_pred = v->acpred_plane[mb_pos] = get_bits1(gb);

        GET_MQUANT();

        s->current_picture.qscale_table[mb_pos] = mquant;

        /* Set DC scale - y and c use the same (not sure if necessary here) */

        s->y_dc_scale = s->y_dc_scale_table[mquant];

        s->c_dc_scale = s->c_dc_scale_table[mquant];

        dst_idx = 0;

        for (i = 0; i < 6; i++) {

            s->dc_val[0][s->block_index[i]] = 0;

            dst_idx += i >> 2;

            val = ((cbp >> (5 - i)) & 1);

            v->mb_type[0][s->block_index[i]] = s->mb_intra;

            v->a_avail = v->c_avail = 0;

            if (i == 2 || i == 3 || !s->first_slice_line)

                v->a_avail = v->mb_type[0][s->block_index[i] - s->block_wrap[i]];

            if (i == 1 || i == 3 || s->mb_x)

                v->c_avail = v->mb_type[0][s->block_index[i] - 1];



            vc1_decode_intra_block(v, s->block[i], i, val, mquant,

                                   (i & 4) ? v->codingset2 : v->codingset);

            if (i > 3 && (s->flags & CODEC_FLAG_GRAY))

                continue;

            v->vc1dsp.vc1_inv_trans_8x8(s->block[i]);

            if (i < 4) {

                stride_y = s->linesize << fieldtx;

                off = (fieldtx) ? ((i & 1) * 8) + ((i & 2) >> 1) * s->linesize : (i & 1) * 8 + 4 * (i & 2) * s->linesize;

            } else {

                stride_y = s->uvlinesize;

                off = 0;

            }

            s->idsp.put_signed_pixels_clamped(s->block[i],

                                              s->dest[dst_idx] + off,

                                              stride_y);

        }

    } else {

        s->mb_intra = v->is_intra[s->mb_x] = 0;

        if (!direct) {

            if (skipped || !s->mb_intra) {

                bmvtype = decode012(gb);

                switch (bmvtype) {

                case 0:

                    bmvtype = (v->bfraction >= (B_FRACTION_DEN/2)) ? BMV_TYPE_BACKWARD : BMV_TYPE_FORWARD;

                    break;

                case 1:

                    bmvtype = (v->bfraction >= (B_FRACTION_DEN/2)) ? BMV_TYPE_FORWARD : BMV_TYPE_BACKWARD;

                    break;

                case 2:

                    bmvtype  = BMV_TYPE_INTERPOLATED;

                }

            }



            if (twomv && bmvtype != BMV_TYPE_INTERPOLATED)

                mvsw = get_bits1(gb);

        }



        if (!skipped) { // inter MB

            mb_has_coeffs = ff_vc1_mbmode_intfrp[0][idx_mbmode][3];

            if (mb_has_coeffs)

                cbp = 1 + get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

            if (!direct) {

                if (bmvtype == BMV_TYPE_INTERPOLATED && twomv) {

                    v->fourmvbp = get_vlc2(gb, v->fourmvbp_vlc->table, VC1_4MV_BLOCK_PATTERN_VLC_BITS, 1);

                } else if (bmvtype == BMV_TYPE_INTERPOLATED || twomv) {

                    v->twomvbp = get_vlc2(gb, v->twomvbp_vlc->table, VC1_2MV_BLOCK_PATTERN_VLC_BITS, 1);

                }

            }



            for (i = 0; i < 6; i++)

                v->mb_type[0][s->block_index[i]] = 0;

            fieldtx = v->fieldtx_plane[mb_pos] = ff_vc1_mbmode_intfrp[0][idx_mbmode][1];

            /* for all motion vector read MVDATA and motion compensate each block */

            dst_idx = 0;

            if (direct) {

                if (twomv) {

                    for (i = 0; i < 4; i++) {

                        ff_vc1_mc_4mv_luma(v, i, 0, 0);

                        ff_vc1_mc_4mv_luma(v, i, 1, 1);

                    }

                    ff_vc1_mc_4mv_chroma4(v, 0, 0, 0);

                    ff_vc1_mc_4mv_chroma4(v, 1, 1, 1);

                } else {

                    ff_vc1_mc_1mv(v, 0);

                    ff_vc1_interp_mc(v);

                }

            } else if (twomv && bmvtype == BMV_TYPE_INTERPOLATED) {

                mvbp = v->fourmvbp;

                for (i = 0; i < 4; i++) {

                    dir = i==1 || i==3;

                    dmv_x = dmv_y = 0;

                    val = ((mvbp >> (3 - i)) & 1);

                    if (val)

                        get_mvdata_interlaced(v, &dmv_x, &dmv_y, 0);

                    j = i > 1 ? 2 : 0;

                    ff_vc1_pred_mv_intfr(v, j, dmv_x, dmv_y, 2, v->range_x, v->range_y, v->mb_type[0], dir);

                    ff_vc1_mc_4mv_luma(v, j, dir, dir);

                    ff_vc1_mc_4mv_luma(v, j+1, dir, dir);

                }



                ff_vc1_mc_4mv_chroma4(v, 0, 0, 0);

                ff_vc1_mc_4mv_chroma4(v, 1, 1, 1);

            } else if (bmvtype == BMV_TYPE_INTERPOLATED) {

                mvbp = v->twomvbp;

                dmv_x = dmv_y = 0;

                if (mvbp & 2)

                    get_mvdata_interlaced(v, &dmv_x, &dmv_y, 0);



                ff_vc1_pred_mv_intfr(v, 0, dmv_x, dmv_y, 1, v->range_x, v->range_y, v->mb_type[0], 0);

                ff_vc1_mc_1mv(v, 0);



                dmv_x = dmv_y = 0;

                if (mvbp & 1)

                    get_mvdata_interlaced(v, &dmv_x, &dmv_y, 0);



                ff_vc1_pred_mv_intfr(v, 0, dmv_x, dmv_y, 1, v->range_x, v->range_y, v->mb_type[0], 1);

                ff_vc1_interp_mc(v);

            } else if (twomv) {

                dir = bmvtype == BMV_TYPE_BACKWARD;

                dir2 = dir;

                if (mvsw)

                    dir2 = !dir;

                mvbp = v->twomvbp;

                dmv_x = dmv_y = 0;

                if (mvbp & 2)

                    get_mvdata_interlaced(v, &dmv_x, &dmv_y, 0);

                ff_vc1_pred_mv_intfr(v, 0, dmv_x, dmv_y, 2, v->range_x, v->range_y, v->mb_type[0], dir);



                dmv_x = dmv_y = 0;

                if (mvbp & 1)

                    get_mvdata_interlaced(v, &dmv_x, &dmv_y, 0);

                ff_vc1_pred_mv_intfr(v, 2, dmv_x, dmv_y, 2, v->range_x, v->range_y, v->mb_type[0], dir2);



                if (mvsw) {

                    for (i = 0; i < 2; i++) {

                        s->mv[dir][i+2][0] = s->mv[dir][i][0] = s->current_picture.motion_val[dir][s->block_index[i+2]][0] = s->current_picture.motion_val[dir][s->block_index[i]][0];

                        s->mv[dir][i+2][1] = s->mv[dir][i][1] = s->current_picture.motion_val[dir][s->block_index[i+2]][1] = s->current_picture.motion_val[dir][s->block_index[i]][1];

                        s->mv[dir2][i+2][0] = s->mv[dir2][i][0] = s->current_picture.motion_val[dir2][s->block_index[i]][0] = s->current_picture.motion_val[dir2][s->block_index[i+2]][0];

                        s->mv[dir2][i+2][1] = s->mv[dir2][i][1] = s->current_picture.motion_val[dir2][s->block_index[i]][1] = s->current_picture.motion_val[dir2][s->block_index[i+2]][1];

                    }

                } else {

                    ff_vc1_pred_mv_intfr(v, 0, 0, 0, 2, v->range_x, v->range_y, v->mb_type[0], !dir);

                    ff_vc1_pred_mv_intfr(v, 2, 0, 0, 2, v->range_x, v->range_y, v->mb_type[0], !dir);

                }



                ff_vc1_mc_4mv_luma(v, 0, dir, 0);

                ff_vc1_mc_4mv_luma(v, 1, dir, 0);

                ff_vc1_mc_4mv_luma(v, 2, dir2, 0);

                ff_vc1_mc_4mv_luma(v, 3, dir2, 0);

                ff_vc1_mc_4mv_chroma4(v, dir, dir2, 0);

            } else {

                dir = bmvtype == BMV_TYPE_BACKWARD;



                mvbp = ff_vc1_mbmode_intfrp[0][idx_mbmode][2];

                dmv_x = dmv_y = 0;

                if (mvbp)

                    get_mvdata_interlaced(v, &dmv_x, &dmv_y, 0);



                ff_vc1_pred_mv_intfr(v, 0, dmv_x, dmv_y, 1, v->range_x, v->range_y, v->mb_type[0], dir);

                v->blk_mv_type[s->block_index[0]] = 1;

                v->blk_mv_type[s->block_index[1]] = 1;

                v->blk_mv_type[s->block_index[2]] = 1;

                v->blk_mv_type[s->block_index[3]] = 1;

                ff_vc1_pred_mv_intfr(v, 0, 0, 0, 2, v->range_x, v->range_y, 0, !dir);

                for (i = 0; i < 2; i++) {

                    s->mv[!dir][i+2][0] = s->mv[!dir][i][0] = s->current_picture.motion_val[!dir][s->block_index[i+2]][0] = s->current_picture.motion_val[!dir][s->block_index[i]][0];

                    s->mv[!dir][i+2][1] = s->mv[!dir][i][1] = s->current_picture.motion_val[!dir][s->block_index[i+2]][1] = s->current_picture.motion_val[!dir][s->block_index[i]][1];

                }

                ff_vc1_mc_1mv(v, dir);

            }



            if (cbp)

                GET_MQUANT();  // p. 227

            s->current_picture.qscale_table[mb_pos] = mquant;

            if (!v->ttmbf && cbp)

                ttmb = get_vlc2(gb, ff_vc1_ttmb_vlc[v->tt_index].table, VC1_TTMB_VLC_BITS, 2);

            for (i = 0; i < 6; i++) {

                s->dc_val[0][s->block_index[i]] = 0;

                dst_idx += i >> 2;

                val = ((cbp >> (5 - i)) & 1);

                if (!fieldtx)

                    off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * s->linesize);

                else

                    off = (i & 4) ? 0 : ((i & 1) * 8 + ((i > 1) * s->linesize));

                if (val) {

                    pat = vc1_decode_p_block(v, s->block[i], i, mquant, ttmb,

                                             first_block, s->dest[dst_idx] + off,

                                             (i & 4) ? s->uvlinesize : (s->linesize << fieldtx),

                                             (i & 4) && (s->flags & CODEC_FLAG_GRAY), &block_tt);

                    block_cbp |= pat << (i << 2);

                    if (!v->ttmbf && ttmb < 8)

                        ttmb = -1;

                    first_block = 0;

                }

            }



        } else { // skipped

            dir = 0;

            for (i = 0; i < 6; i++) {

                v->mb_type[0][s->block_index[i]] = 0;

                s->dc_val[0][s->block_index[i]] = 0;

            }

            s->current_picture.mb_type[mb_pos]      = MB_TYPE_SKIP;

            s->current_picture.qscale_table[mb_pos] = 0;

            v->blk_mv_type[s->block_index[0]] = 0;

            v->blk_mv_type[s->block_index[1]] = 0;

            v->blk_mv_type[s->block_index[2]] = 0;

            v->blk_mv_type[s->block_index[3]] = 0;



            if (!direct) {

                if (bmvtype == BMV_TYPE_INTERPOLATED) {

                    ff_vc1_pred_mv_intfr(v, 0, 0, 0, 1, v->range_x, v->range_y, v->mb_type[0], 0);

                    ff_vc1_pred_mv_intfr(v, 0, 0, 0, 1, v->range_x, v->range_y, v->mb_type[0], 1);

                } else {

                    dir = bmvtype == BMV_TYPE_BACKWARD;

                    ff_vc1_pred_mv_intfr(v, 0, 0, 0, 1, v->range_x, v->range_y, v->mb_type[0], dir);

                    if (mvsw) {

                        int dir2 = dir;

                        if (mvsw)

                            dir2 = !dir;

                        for (i = 0; i < 2; i++) {

                            s->mv[dir][i+2][0] = s->mv[dir][i][0] = s->current_picture.motion_val[dir][s->block_index[i+2]][0] = s->current_picture.motion_val[dir][s->block_index[i]][0];

                            s->mv[dir][i+2][1] = s->mv[dir][i][1] = s->current_picture.motion_val[dir][s->block_index[i+2]][1] = s->current_picture.motion_val[dir][s->block_index[i]][1];

                            s->mv[dir2][i+2][0] = s->mv[dir2][i][0] = s->current_picture.motion_val[dir2][s->block_index[i]][0] = s->current_picture.motion_val[dir2][s->block_index[i+2]][0];

                            s->mv[dir2][i+2][1] = s->mv[dir2][i][1] = s->current_picture.motion_val[dir2][s->block_index[i]][1] = s->current_picture.motion_val[dir2][s->block_index[i+2]][1];

                        }

                    } else {

                        v->blk_mv_type[s->block_index[0]] = 1;

                        v->blk_mv_type[s->block_index[1]] = 1;

                        v->blk_mv_type[s->block_index[2]] = 1;

                        v->blk_mv_type[s->block_index[3]] = 1;

                        ff_vc1_pred_mv_intfr(v, 0, 0, 0, 2, v->range_x, v->range_y, 0, !dir);

                        for (i = 0; i < 2; i++) {

                            s->mv[!dir][i+2][0] = s->mv[!dir][i][0] = s->current_picture.motion_val[!dir][s->block_index[i+2]][0] = s->current_picture.motion_val[!dir][s->block_index[i]][0];

                            s->mv[!dir][i+2][1] = s->mv[!dir][i][1] = s->current_picture.motion_val[!dir][s->block_index[i+2]][1] = s->current_picture.motion_val[!dir][s->block_index[i]][1];

                        }

                    }

                }

            }



            ff_vc1_mc_1mv(v, dir);

            if (direct || bmvtype == BMV_TYPE_INTERPOLATED) {

                ff_vc1_interp_mc(v);

            }

        }

    }

    if (s->mb_x == s->mb_width - 1)

        memmove(v->is_intra_base, v->is_intra, sizeof(v->is_intra_base[0]) * s->mb_stride);

    v->cbp[s->mb_x]      = block_cbp;

    v->ttblk[s->mb_x]    = block_tt;

    return 0;

}
