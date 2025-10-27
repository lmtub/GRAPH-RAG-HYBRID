void ff_mpeg4_encode_mb(MpegEncContext *s, int16_t block[6][64],

                        int motion_x, int motion_y)

{

    int cbpc, cbpy, pred_x, pred_y;

    PutBitContext *const pb2    = s->data_partitioning ? &s->pb2 : &s->pb;

    PutBitContext *const tex_pb = s->data_partitioning && s->pict_type != AV_PICTURE_TYPE_B ? &s->tex_pb : &s->pb;

    PutBitContext *const dc_pb  = s->data_partitioning && s->pict_type != AV_PICTURE_TYPE_I ? &s->pb2 : &s->pb;

    const int interleaved_stats = (s->flags & CODEC_FLAG_PASS1) && !s->data_partitioning ? 1 : 0;



    if (!s->mb_intra) {

        int i, cbp;



        if (s->pict_type == AV_PICTURE_TYPE_B) {

            /* convert from mv_dir to type */

            static const int mb_type_table[8] = { -1, 3, 2, 1, -1, -1, -1, 0 };

            int mb_type = mb_type_table[s->mv_dir];



            if (s->mb_x == 0) {

                for (i = 0; i < 2; i++)

                    s->last_mv[i][0][0] =

                    s->last_mv[i][0][1] =

                    s->last_mv[i][1][0] =

                    s->last_mv[i][1][1] = 0;

            }



            assert(s->dquant >= -2 && s->dquant <= 2);

            assert((s->dquant & 1) == 0);

            assert(mb_type >= 0);



            /* nothing to do if this MB was skipped in the next P Frame */

            if (s->next_picture.mbskip_table[s->mb_y * s->mb_stride + s->mb_x]) {  // FIXME avoid DCT & ...

                s->skip_count++;

                s->mv[0][0][0] =

                s->mv[0][0][1] =

                s->mv[1][0][0] =

                s->mv[1][0][1] = 0;

                s->mv_dir  = MV_DIR_FORWARD;  // doesn't matter

                s->qscale -= s->dquant;

//                s->mb_skipped = 1;



                return;

            }



            cbp = get_b_cbp(s, block, motion_x, motion_y, mb_type);



            if ((cbp | motion_x | motion_y | mb_type) == 0) {

                /* direct MB with MV={0,0} */

                assert(s->dquant == 0);



                put_bits(&s->pb, 1, 1); /* mb not coded modb1=1 */



                if (interleaved_stats) {

                    s->misc_bits++;

                    s->last_bits++;

                }

                s->skip_count++;

                return;

            }



            put_bits(&s->pb, 1, 0);            /* mb coded modb1=0 */

            put_bits(&s->pb, 1, cbp ? 0 : 1);  /* modb2 */ // FIXME merge

            put_bits(&s->pb, mb_type + 1, 1);  // this table is so simple that we don't need it :)

            if (cbp)

                put_bits(&s->pb, 6, cbp);



            if (cbp && mb_type) {

                if (s->dquant)

                    put_bits(&s->pb, 2, (s->dquant >> 2) + 3);

                else

                    put_bits(&s->pb, 1, 0);

            } else

                s->qscale -= s->dquant;



            if (!s->progressive_sequence) {

                if (cbp)

                    put_bits(&s->pb, 1, s->interlaced_dct);

                if (mb_type)                  // not direct mode

                    put_bits(&s->pb, 1, s->mv_type == MV_TYPE_FIELD);

            }



            if (interleaved_stats)

                s->misc_bits += get_bits_diff(s);



            if (!mb_type) {

                assert(s->mv_dir & MV_DIRECT);

                ff_h263_encode_motion_vector(s, motion_x, motion_y, 1);

                s->b_count++;

                s->f_count++;

            } else {

                assert(mb_type > 0 && mb_type < 4);

                if (s->mv_type != MV_TYPE_FIELD) {

                    if (s->mv_dir & MV_DIR_FORWARD) {

                        ff_h263_encode_motion_vector(s,

                                                     s->mv[0][0][0] - s->last_mv[0][0][0],

                                                     s->mv[0][0][1] - s->last_mv[0][0][1],

                                                     s->f_code);

                        s->last_mv[0][0][0] =

                        s->last_mv[0][1][0] = s->mv[0][0][0];

                        s->last_mv[0][0][1] =

                        s->last_mv[0][1][1] = s->mv[0][0][1];

                        s->f_count++;

                    }

                    if (s->mv_dir & MV_DIR_BACKWARD) {

                        ff_h263_encode_motion_vector(s,

                                                     s->mv[1][0][0] - s->last_mv[1][0][0],

                                                     s->mv[1][0][1] - s->last_mv[1][0][1],

                                                     s->b_code);

                        s->last_mv[1][0][0] =

                        s->last_mv[1][1][0] = s->mv[1][0][0];

                        s->last_mv[1][0][1] =

                        s->last_mv[1][1][1] = s->mv[1][0][1];

                        s->b_count++;

                    }

                } else {

                    if (s->mv_dir & MV_DIR_FORWARD) {

                        put_bits(&s->pb, 1, s->field_select[0][0]);

                        put_bits(&s->pb, 1, s->field_select[0][1]);

                    }

                    if (s->mv_dir & MV_DIR_BACKWARD) {

                        put_bits(&s->pb, 1, s->field_select[1][0]);

                        put_bits(&s->pb, 1, s->field_select[1][1]);

                    }

                    if (s->mv_dir & MV_DIR_FORWARD) {

                        for (i = 0; i < 2; i++) {

                            ff_h263_encode_motion_vector(s,

                                                         s->mv[0][i][0] - s->last_mv[0][i][0],

                                                         s->mv[0][i][1] - s->last_mv[0][i][1] / 2,

                                                         s->f_code);

                            s->last_mv[0][i][0] = s->mv[0][i][0];

                            s->last_mv[0][i][1] = s->mv[0][i][1] * 2;

                        }

                        s->f_count++;

                    }

                    if (s->mv_dir & MV_DIR_BACKWARD) {

                        for (i = 0; i < 2; i++) {

                            ff_h263_encode_motion_vector(s,

                                                         s->mv[1][i][0] - s->last_mv[1][i][0],

                                                         s->mv[1][i][1] - s->last_mv[1][i][1] / 2,

                                                         s->b_code);

                            s->last_mv[1][i][0] = s->mv[1][i][0];

                            s->last_mv[1][i][1] = s->mv[1][i][1] * 2;

                        }

                        s->b_count++;

                    }

                }

            }



            if (interleaved_stats)

                s->mv_bits += get_bits_diff(s);



            mpeg4_encode_blocks(s, block, NULL, NULL, NULL, &s->pb);



            if (interleaved_stats)

                s->p_tex_bits += get_bits_diff(s);

        } else { /* s->pict_type==AV_PICTURE_TYPE_B */

            cbp = get_p_cbp(s, block, motion_x, motion_y);



            if ((cbp | motion_x | motion_y | s->dquant) == 0 &&

                s->mv_type == MV_TYPE_16X16) {

                /* check if the B frames can skip it too, as we must skip it

                 * if we skip here why didn't they just compress

                 * the skip-mb bits instead of reusing them ?! */

                if (s->max_b_frames > 0) {

                    int i;

                    int x, y, offset;

                    uint8_t *p_pic;



                    x = s->mb_x * 16;

                    y = s->mb_y * 16;

                    if (x + 16 > s->width)

                        x = s->width - 16;

                    if (y + 16 > s->height)

                        y = s->height - 16;



                    offset = x + y * s->linesize;

                    p_pic  = s->new_picture.f.data[0] + offset;



                    s->mb_skipped = 1;

                    for (i = 0; i < s->max_b_frames; i++) {

                        uint8_t *b_pic;

                        int diff;

                        Picture *pic = s->reordered_input_picture[i + 1];



                        if (!pic || pic->f.pict_type != AV_PICTURE_TYPE_B)

                            break;



                        b_pic = pic->f.data[0] + offset;

                        if (!pic->shared)

                            b_pic += INPLACE_OFFSET;

                        diff = s->dsp.sad[0](NULL, p_pic, b_pic, s->linesize, 16);

                        if (diff > s->qscale * 70) {  // FIXME check that 70 is optimal

                            s->mb_skipped = 0;

                            break;

                        }

                    }

                } else

                    s->mb_skipped = 1;



                if (s->mb_skipped == 1) {

                    /* skip macroblock */

                    put_bits(&s->pb, 1, 1);



                    if (interleaved_stats) {

                        s->misc_bits++;

                        s->last_bits++;

                    }

                    s->skip_count++;



                    return;

                }

            }



            put_bits(&s->pb, 1, 0);     /* mb coded */

            cbpc  = cbp & 3;

            cbpy  = cbp >> 2;

            cbpy ^= 0xf;

            if (s->mv_type == MV_TYPE_16X16) {

                if (s->dquant)

                    cbpc += 8;

                put_bits(&s->pb,

                         ff_h263_inter_MCBPC_bits[cbpc],

                         ff_h263_inter_MCBPC_code[cbpc]);



                put_bits(pb2, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);

                if (s->dquant)

                    put_bits(pb2, 2, dquant_code[s->dquant + 2]);



                if (!s->progressive_sequence) {

                    if (cbp)

                        put_bits(pb2, 1, s->interlaced_dct);

                    put_bits(pb2, 1, 0);

                }



                if (interleaved_stats)

                    s->misc_bits += get_bits_diff(s);



                /* motion vectors: 16x16 mode */

                ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);



                ff_h263_encode_motion_vector(s,

                                             motion_x - pred_x,

                                             motion_y - pred_y,

                                             s->f_code);

            } else if (s->mv_type == MV_TYPE_FIELD) {

                if (s->dquant)

                    cbpc += 8;

                put_bits(&s->pb,

                         ff_h263_inter_MCBPC_bits[cbpc],

                         ff_h263_inter_MCBPC_code[cbpc]);



                put_bits(pb2, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);

                if (s->dquant)

                    put_bits(pb2, 2, dquant_code[s->dquant + 2]);



                assert(!s->progressive_sequence);

                if (cbp)

                    put_bits(pb2, 1, s->interlaced_dct);

                put_bits(pb2, 1, 1);



                if (interleaved_stats)

                    s->misc_bits += get_bits_diff(s);



                /* motion vectors: 16x8 interlaced mode */

                ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);

                pred_y /= 2;



                put_bits(&s->pb, 1, s->field_select[0][0]);

                put_bits(&s->pb, 1, s->field_select[0][1]);



                ff_h263_encode_motion_vector(s,

                                             s->mv[0][0][0] - pred_x,

                                             s->mv[0][0][1] - pred_y,

                                             s->f_code);

                ff_h263_encode_motion_vector(s,

                                             s->mv[0][1][0] - pred_x,

                                             s->mv[0][1][1] - pred_y,

                                             s->f_code);

            } else {

                assert(s->mv_type == MV_TYPE_8X8);

                put_bits(&s->pb,

                         ff_h263_inter_MCBPC_bits[cbpc + 16],

                         ff_h263_inter_MCBPC_code[cbpc + 16]);

                put_bits(pb2, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);



                if (!s->progressive_sequence && cbp)

                    put_bits(pb2, 1, s->interlaced_dct);



                if (interleaved_stats)

                    s->misc_bits += get_bits_diff(s);



                for (i = 0; i < 4; i++) {

                    /* motion vectors: 8x8 mode*/

                    ff_h263_pred_motion(s, i, 0, &pred_x, &pred_y);



                    ff_h263_encode_motion_vector(s,

                                                 s->current_picture.motion_val[0][s->block_index[i]][0] - pred_x,

                                                 s->current_picture.motion_val[0][s->block_index[i]][1] - pred_y,

                                                 s->f_code);

                }

            }



            if (interleaved_stats)

                s->mv_bits += get_bits_diff(s);



            mpeg4_encode_blocks(s, block, NULL, NULL, NULL, tex_pb);



            if (interleaved_stats)

                s->p_tex_bits += get_bits_diff(s);



            s->f_count++;

        }

    } else {

        int cbp;

        int dc_diff[6];  // dc values with the dc prediction subtracted

        int dir[6];      // prediction direction

        int zigzag_last_index[6];

        uint8_t *scan_table[6];

        int i;



        for (i = 0; i < 6; i++)

            dc_diff[i] = ff_mpeg4_pred_dc(s, i, block[i][0], &dir[i], 1);



        if (s->flags & CODEC_FLAG_AC_PRED) {

            s->ac_pred = decide_ac_pred(s, block, dir, scan_table, zigzag_last_index);

        } else {

            for (i = 0; i < 6; i++)

                scan_table[i] = s->intra_scantable.permutated;

        }



        /* compute cbp */

        cbp = 0;

        for (i = 0; i < 6; i++)

            if (s->block_last_index[i] >= 1)

                cbp |= 1 << (5 - i);



        cbpc = cbp & 3;

        if (s->pict_type == AV_PICTURE_TYPE_I) {

            if (s->dquant)

                cbpc += 4;

            put_bits(&s->pb,

                     ff_h263_intra_MCBPC_bits[cbpc],

                     ff_h263_intra_MCBPC_code[cbpc]);

        } else {

            if (s->dquant)

                cbpc += 8;

            put_bits(&s->pb, 1, 0);     /* mb coded */

            put_bits(&s->pb,

                     ff_h263_inter_MCBPC_bits[cbpc + 4],

                     ff_h263_inter_MCBPC_code[cbpc + 4]);

        }

        put_bits(pb2, 1, s->ac_pred);

        cbpy = cbp >> 2;

        put_bits(pb2, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);

        if (s->dquant)

            put_bits(dc_pb, 2, dquant_code[s->dquant + 2]);



        if (!s->progressive_sequence)

            put_bits(dc_pb, 1, s->interlaced_dct);



        if (interleaved_stats)

            s->misc_bits += get_bits_diff(s);



        mpeg4_encode_blocks(s, block, dc_diff, scan_table, dc_pb, tex_pb);



        if (interleaved_stats)

            s->i_tex_bits += get_bits_diff(s);

        s->i_count++;



        /* restore ac coeffs & last_index stuff

         * if we messed them up with the prediction */

        if (s->ac_pred)

            restore_ac_coeffs(s, block, dir, scan_table, zigzag_last_index);

    }

}
