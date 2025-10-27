static int vc1_decode_p_mb(VC1Context *v, DCTELEM block[6][64])

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &s->gb;

    int i, j;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int cbp; /* cbp decoding stuff */

    int mqdiff, mquant; /* MB quantization */

    int ttmb = v->ttfrm; /* MB Transform type */

    int status;



    static const int size_table[6] = { 0, 2, 3, 4, 5, 8 },

      offset_table[6] = { 0, 1, 3, 7, 15, 31 };

    int mb_has_coeffs = 1; /* last_flag */

    int dmv_x, dmv_y; /* Differential MV components */

    int index, index1; /* LUT indices */

    int val, sign; /* temp values */

    int first_block = 1;

    int dst_idx, off;

    int skipped, fourmv;



    mquant = v->pq; /* Loosy initialization */



    if (v->mv_type_is_raw)

        fourmv = get_bits1(gb);

    else

        fourmv = v->mv_type_mb_plane[mb_pos];

    if (v->skip_is_raw)

        skipped = get_bits1(gb);

    else

        skipped = v->s.mbskip_table[mb_pos];



    s->dsp.clear_blocks(s->block[0]);



    if (!fourmv) /* 1MV mode */

    {

        if (!skipped)

        {

            GET_MVDATA(dmv_x, dmv_y);



            s->current_picture.mb_type[mb_pos] = s->mb_intra ? MB_TYPE_INTRA : MB_TYPE_16x16;

            vc1_pred_mv(s, 0, dmv_x, dmv_y, 1, v->range_x, v->range_y, v->mb_type[0]);



            /* FIXME Set DC val for inter block ? */

            if (s->mb_intra && !mb_has_coeffs)

            {

                GET_MQUANT();

                s->ac_pred = get_bits(gb, 1);

                cbp = 0;

            }

            else if (mb_has_coeffs)

            {

                if (s->mb_intra) s->ac_pred = get_bits(gb, 1);

                cbp = get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

                GET_MQUANT();

            }

            else

            {

                mquant = v->pq;

                cbp = 0;

            }

            s->current_picture.qscale_table[mb_pos] = mquant;



            if (!v->ttmbf && !s->mb_intra && mb_has_coeffs)

                ttmb = get_vlc2(gb, vc1_ttmb_vlc[v->tt_index].table,

                                VC1_TTMB_VLC_BITS, 2);

            if(!s->mb_intra) vc1_mc_1mv(v);

            dst_idx = 0;

            for (i=0; i<6; i++)

            {

                s->dc_val[0][s->block_index[i]] = 0;

                dst_idx += i >> 2;

                val = ((cbp >> (5 - i)) & 1);

                off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * s->linesize);

                v->mb_type[0][s->block_index[i]] = s->mb_intra;

                if(s->mb_intra) {

                    /* check if prediction blocks A and C are available */

                    v->a_avail = v->c_avail = 0;

                    if(i == 2 || i == 3 || s->mb_y)

                        v->a_avail = v->mb_type[0][s->block_index[i] - s->block_wrap[i]];

                    if(i == 1 || i == 3 || s->mb_x)

                        v->c_avail = v->mb_type[0][s->block_index[i] - 1];



                    vc1_decode_intra_block(v, block[i], i, val, mquant, (i&4)?v->codingset2:v->codingset);

                    vc1_inv_trans(block[i], 8, 8);

                    for(j = 0; j < 64; j++) block[i][j] += 128;

                    s->dsp.put_pixels_clamped(block[i], s->dest[dst_idx] + off, s->linesize >> ((i & 4) >> 2));

                    /* TODO: proper loop filtering */

                    if(v->pq >= 9 && v->overlap) {

                        if(v->a_avail)

                            s->dsp.h263_v_loop_filter(s->dest[dst_idx] + off, s->linesize >> ((i & 4) >> 2), s->y_dc_scale);

                        if(v->c_avail)

                            s->dsp.h263_h_loop_filter(s->dest[dst_idx] + off, s->linesize >> ((i & 4) >> 2), s->y_dc_scale);

                    }

                } else if(val) {

                    vc1_decode_p_block(v, block[i], i, mquant, ttmb, first_block);

                    if(!v->ttmbf && ttmb < 8) ttmb = -1;

                    first_block = 0;

                    s->dsp.add_pixels_clamped(block[i], s->dest[dst_idx] + off, (i&4)?s->uvlinesize:s->linesize);

                }

            }

        }

        else //Skipped

        {

            s->mb_intra = 0;

            for(i = 0; i < 6; i++) v->mb_type[0][s->block_index[i]] = 0;

            s->current_picture.mb_type[mb_pos] = MB_TYPE_SKIP;

            s->current_picture.qscale_table[mb_pos] = 0;

            vc1_pred_mv(s, 0, 0, 0, 1, v->range_x, v->range_y, v->mb_type[0]);

            vc1_mc_1mv(v);

            return 0;

        }

    } //1MV mode

    else //4MV mode

    {

        if (!skipped /* unskipped MB */)

        {

            int intra_count = 0, coded_inter = 0;

            int is_intra[6], is_coded[6];

            /* Get CBPCY */

            cbp = get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

            for (i=0; i<6; i++)

            {

                val = ((cbp >> (5 - i)) & 1);

                s->dc_val[0][s->block_index[i]] = 0;

                s->mb_intra = 0;

                if(i < 4) {

                    dmv_x = dmv_y = 0;

                    s->mb_intra = 0;

                    mb_has_coeffs = 0;

                    if(val) {

                        GET_MVDATA(dmv_x, dmv_y);

                    }

                    vc1_pred_mv(s, i, dmv_x, dmv_y, 0, v->range_x, v->range_y, v->mb_type[0]);

                    if(!s->mb_intra) vc1_mc_4mv_luma(v, i);

                    intra_count += s->mb_intra;

                    is_intra[i] = s->mb_intra;

                    is_coded[i] = mb_has_coeffs;

                }

                if(i&4){

                    is_intra[i] = (intra_count >= 3);

                    is_coded[i] = val;

                }

                if(i == 4) vc1_mc_4mv_chroma(v);

                v->mb_type[0][s->block_index[i]] = is_intra[i];

                if(!coded_inter) coded_inter = !is_intra[i] & is_coded[i];

            }

            dst_idx = 0;

            GET_MQUANT();

            s->current_picture.qscale_table[mb_pos] = mquant;

            /* test if block is intra and has pred */

            {

                int intrapred = 0;

                for(i=0; i<6; i++)

                    if(is_intra[i]) {

                        if(v->mb_type[0][s->block_index[i] - s->block_wrap[i]] || v->mb_type[0][s->block_index[i] - 1]) {

                            intrapred = 1;

                            break;

                        }

                    }

                if(intrapred)s->ac_pred = get_bits(gb, 1);

                else s->ac_pred = 0;

            }

            if (!v->ttmbf && coded_inter)

                ttmb = get_vlc2(gb, vc1_ttmb_vlc[v->tt_index].table, VC1_TTMB_VLC_BITS, 12);

            for (i=0; i<6; i++)

            {

                dst_idx += i >> 2;

                off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * s->linesize);

                s->mb_intra = is_intra[i];

                if (is_intra[i]) {

                    /* check if prediction blocks A and C are available */

                    v->a_avail = v->c_avail = 0;

                    if(i == 2 || i == 3 || s->mb_y)

                        v->a_avail = v->mb_type[0][s->block_index[i] - s->block_wrap[i]];

                    if(i == 1 || i == 3 || s->mb_x)

                        v->c_avail = v->mb_type[0][s->block_index[i] - 1];



                    vc1_decode_intra_block(v, s->block[i], i, is_coded[i], mquant, (i&4)?v->codingset2:v->codingset);

                    vc1_inv_trans(block[i], 8, 8);

                    for(j = 0; j < 64; j++) block[i][j] += 128;

                    s->dsp.put_pixels_clamped(s->block[i], s->dest[dst_idx] + off, (i&4)?s->uvlinesize:s->linesize);

                    /* TODO: proper loop filtering */

                    if(v->pq >= 9 && v->overlap) {

                        if(v->a_avail)

                            s->dsp.h263_v_loop_filter(s->dest[dst_idx] + off, s->linesize >> ((i & 4) >> 2), s->y_dc_scale);

                        if(v->c_avail)

                            s->dsp.h263_h_loop_filter(s->dest[dst_idx] + off, s->linesize >> ((i & 4) >> 2), s->y_dc_scale);

                    }

                } else if(is_coded[i]) {

                    status = vc1_decode_p_block(v, s->block[i], i, mquant, ttmb, first_block);

                    if(!v->ttmbf && ttmb < 8) ttmb = -1;

                    first_block = 0;

                    s->dsp.add_pixels_clamped(s->block[i], s->dest[dst_idx] + off, (i&4)?s->uvlinesize:s->linesize);

                }

            }

            return status;

        }

        else //Skipped MB

        {

            s->mb_intra = 0;

            for (i=0; i<6; i++) v->mb_type[0][s->block_index[i]] = 0;

            for (i=0; i<4; i++)

            {

                vc1_pred_mv(s, i, 0, 0, 0, v->range_x, v->range_y, v->mb_type[0]);

                vc1_mc_4mv_luma(v, i);

            }

            vc1_mc_4mv_chroma(v);

            s->current_picture.qscale_table[mb_pos] = 0;

            return 0;

        }

    }



    /* Should never happen */

    return -1;

}
