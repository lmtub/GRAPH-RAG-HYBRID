static int mpeg_decode_mb(MpegEncContext *s, DCTELEM block[12][64])

{

    int i, j, k, cbp, val, mb_type, motion_type;

    const int mb_block_count = 4 + (1 << s->chroma_format);



    av_dlog(s->avctx, "decode_mb: x=%d y=%d\n", s->mb_x, s->mb_y);



    assert(s->mb_skipped == 0);



    if (s->mb_skip_run-- != 0) {

        if (s->pict_type == AV_PICTURE_TYPE_P) {

            s->mb_skipped = 1;

            s->current_picture.f.mb_type[s->mb_x + s->mb_y * s->mb_stride] = MB_TYPE_SKIP | MB_TYPE_L0 | MB_TYPE_16x16;

        } else {

            int mb_type;



            if (s->mb_x)

                mb_type = s->current_picture.f.mb_type[s->mb_x + s->mb_y * s->mb_stride - 1];

            else

                mb_type = s->current_picture.f.mb_type[s->mb_width + (s->mb_y - 1) * s->mb_stride - 1]; // FIXME not sure if this is allowed in MPEG at all

            if (IS_INTRA(mb_type))

                return -1;

            s->current_picture.f.mb_type[s->mb_x + s->mb_y*s->mb_stride] =

                mb_type | MB_TYPE_SKIP;

//            assert(s->current_picture.f.mb_type[s->mb_x + s->mb_y * s->mb_stride - 1] & (MB_TYPE_16x16 | MB_TYPE_16x8));



            if ((s->mv[0][0][0] | s->mv[0][0][1] | s->mv[1][0][0] | s->mv[1][0][1]) == 0)

                s->mb_skipped = 1;

        }



        return 0;

    }



    switch (s->pict_type) {

    default:

    case AV_PICTURE_TYPE_I:

        if (get_bits1(&s->gb) == 0) {

            if (get_bits1(&s->gb) == 0) {

                av_log(s->avctx, AV_LOG_ERROR, "invalid mb type in I Frame at %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }

            mb_type = MB_TYPE_QUANT | MB_TYPE_INTRA;

        } else {

            mb_type = MB_TYPE_INTRA;

        }

        break;

    case AV_PICTURE_TYPE_P:

        mb_type = get_vlc2(&s->gb, mb_ptype_vlc.table, MB_PTYPE_VLC_BITS, 1);

        if (mb_type < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "invalid mb type in P Frame at %d %d\n", s->mb_x, s->mb_y);

            return -1;

        }

        mb_type = ptype2mb_type[mb_type];

        break;

    case AV_PICTURE_TYPE_B:

        mb_type = get_vlc2(&s->gb, mb_btype_vlc.table, MB_BTYPE_VLC_BITS, 1);

        if (mb_type < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "invalid mb type in B Frame at %d %d\n", s->mb_x, s->mb_y);

            return -1;

        }

        mb_type = btype2mb_type[mb_type];

        break;

    }

    av_dlog(s->avctx, "mb_type=%x\n", mb_type);

//    motion_type = 0; /* avoid warning */

    if (IS_INTRA(mb_type)) {

        s->dsp.clear_blocks(s->block[0]);



        if (!s->chroma_y_shift) {

            s->dsp.clear_blocks(s->block[6]);

        }



        /* compute DCT type */

        if (s->picture_structure == PICT_FRAME && // FIXME add an interlaced_dct coded var?

            !s->frame_pred_frame_dct) {

            s->interlaced_dct = get_bits1(&s->gb);

        }



        if (IS_QUANT(mb_type))

            s->qscale = get_qscale(s);



        if (s->concealment_motion_vectors) {

            /* just parse them */

            if (s->picture_structure != PICT_FRAME)

                skip_bits1(&s->gb); /* field select */



            s->mv[0][0][0]= s->last_mv[0][0][0]= s->last_mv[0][1][0] =

                mpeg_decode_motion(s, s->mpeg_f_code[0][0], s->last_mv[0][0][0]);

            s->mv[0][0][1]= s->last_mv[0][0][1]= s->last_mv[0][1][1] =

                mpeg_decode_motion(s, s->mpeg_f_code[0][1], s->last_mv[0][0][1]);



            skip_bits1(&s->gb); /* marker */

        } else

            memset(s->last_mv, 0, sizeof(s->last_mv)); /* reset mv prediction */

        s->mb_intra = 1;

        // if 1, we memcpy blocks in xvmcvideo

        if (CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration > 1) {

            ff_xvmc_pack_pblocks(s, -1); // inter are always full blocks

            if (s->swap_uv) {

                exchange_uv(s);

            }

        }



        if (s->codec_id == CODEC_ID_MPEG2VIDEO) {

            if (s->flags2 & CODEC_FLAG2_FAST) {

                for (i = 0; i < 6; i++) {

                    mpeg2_fast_decode_block_intra(s, *s->pblocks[i], i);

                }

            } else {

                for (i = 0; i < mb_block_count; i++) {

                    if (mpeg2_decode_block_intra(s, *s->pblocks[i], i) < 0)

                        return -1;

                }

            }

        } else {

            for (i = 0; i < 6; i++) {

                if (mpeg1_decode_block_intra(s, *s->pblocks[i], i) < 0)

                    return -1;

            }

        }

    } else {

        if (mb_type & MB_TYPE_ZERO_MV) {

            assert(mb_type & MB_TYPE_CBP);



            s->mv_dir = MV_DIR_FORWARD;

            if (s->picture_structure == PICT_FRAME) {

                if (!s->frame_pred_frame_dct)

                    s->interlaced_dct = get_bits1(&s->gb);

                s->mv_type = MV_TYPE_16X16;

            } else {

                s->mv_type = MV_TYPE_FIELD;

                mb_type |= MB_TYPE_INTERLACED;

                s->field_select[0][0] = s->picture_structure - 1;

            }



            if (IS_QUANT(mb_type))

                s->qscale = get_qscale(s);



            s->last_mv[0][0][0] = 0;

            s->last_mv[0][0][1] = 0;

            s->last_mv[0][1][0] = 0;

            s->last_mv[0][1][1] = 0;

            s->mv[0][0][0] = 0;

            s->mv[0][0][1] = 0;

        } else {

            assert(mb_type & MB_TYPE_L0L1);

            // FIXME decide if MBs in field pictures are MB_TYPE_INTERLACED

            /* get additional motion vector type */

            if (s->frame_pred_frame_dct)

                motion_type = MT_FRAME;

            else {

                motion_type = get_bits(&s->gb, 2);

                if (s->picture_structure == PICT_FRAME && HAS_CBP(mb_type))

                    s->interlaced_dct = get_bits1(&s->gb);

            }



            if (IS_QUANT(mb_type))

                s->qscale = get_qscale(s);



            /* motion vectors */

            s->mv_dir = (mb_type >> 13) & 3;

            av_dlog(s->avctx, "motion_type=%d\n", motion_type);

            switch (motion_type) {

            case MT_FRAME: /* or MT_16X8 */

                if (s->picture_structure == PICT_FRAME) {

                    mb_type |= MB_TYPE_16x16;

                    s->mv_type = MV_TYPE_16X16;

                    for (i = 0; i < 2; i++) {

                        if (USES_LIST(mb_type, i)) {

                            /* MT_FRAME */

                            s->mv[i][0][0]= s->last_mv[i][0][0]= s->last_mv[i][1][0] =

                                mpeg_decode_motion(s, s->mpeg_f_code[i][0], s->last_mv[i][0][0]);

                            s->mv[i][0][1]= s->last_mv[i][0][1]= s->last_mv[i][1][1] =

                                mpeg_decode_motion(s, s->mpeg_f_code[i][1], s->last_mv[i][0][1]);

                            /* full_pel: only for MPEG-1 */

                            if (s->full_pel[i]) {

                                s->mv[i][0][0] <<= 1;

                                s->mv[i][0][1] <<= 1;

                            }

                        }

                    }

                } else {

                    mb_type |= MB_TYPE_16x8 | MB_TYPE_INTERLACED;

                    s->mv_type = MV_TYPE_16X8;

                    for (i = 0; i < 2; i++) {

                        if (USES_LIST(mb_type, i)) {

                            /* MT_16X8 */

                            for (j = 0; j < 2; j++) {

                                s->field_select[i][j] = get_bits1(&s->gb);

                                for (k = 0; k < 2; k++) {

                                    val = mpeg_decode_motion(s, s->mpeg_f_code[i][k],

                                                             s->last_mv[i][j][k]);

                                    s->last_mv[i][j][k] = val;

                                    s->mv[i][j][k]      = val;

                                }

                            }

                        }

                    }

                }

                break;

            case MT_FIELD:

                if(s->progressive_sequence){

                    av_log(s->avctx, AV_LOG_ERROR, "MT_FIELD in progressive_sequence\n");

                    return -1;

                }

                s->mv_type = MV_TYPE_FIELD;

                if (s->picture_structure == PICT_FRAME) {

                    mb_type |= MB_TYPE_16x8 | MB_TYPE_INTERLACED;

                    for (i = 0; i < 2; i++) {

                        if (USES_LIST(mb_type, i)) {

                            for (j = 0; j < 2; j++) {

                                s->field_select[i][j] = get_bits1(&s->gb);

                                val = mpeg_decode_motion(s, s->mpeg_f_code[i][0],

                                                         s->last_mv[i][j][0]);

                                s->last_mv[i][j][0] = val;

                                s->mv[i][j][0]      = val;

                                av_dlog(s->avctx, "fmx=%d\n", val);

                                val = mpeg_decode_motion(s, s->mpeg_f_code[i][1],

                                                         s->last_mv[i][j][1] >> 1);

                                s->last_mv[i][j][1] = val << 1;

                                s->mv[i][j][1]      = val;

                                av_dlog(s->avctx, "fmy=%d\n", val);

                            }

                        }

                    }

                } else {

                    av_assert0(!s->progressive_sequence);

                    mb_type |= MB_TYPE_16x16 | MB_TYPE_INTERLACED;

                    for (i = 0; i < 2; i++) {

                        if (USES_LIST(mb_type, i)) {

                            s->field_select[i][0] = get_bits1(&s->gb);

                            for (k = 0; k < 2; k++) {

                                val = mpeg_decode_motion(s, s->mpeg_f_code[i][k],

                                                         s->last_mv[i][0][k]);

                                s->last_mv[i][0][k] = val;

                                s->last_mv[i][1][k] = val;

                                s->mv[i][0][k]      = val;

                            }

                        }

                    }

                }

                break;

            case MT_DMV:

                if(s->progressive_sequence){

                    av_log(s->avctx, AV_LOG_ERROR, "MT_DMV in progressive_sequence\n");

                    return -1;

                }

                s->mv_type = MV_TYPE_DMV;

                for (i = 0; i < 2; i++) {

                    if (USES_LIST(mb_type, i)) {

                        int dmx, dmy, mx, my, m;

                        const int my_shift = s->picture_structure == PICT_FRAME;



                        mx = mpeg_decode_motion(s, s->mpeg_f_code[i][0],

                                                s->last_mv[i][0][0]);

                        s->last_mv[i][0][0] = mx;

                        s->last_mv[i][1][0] = mx;

                        dmx = get_dmv(s);

                        my  = mpeg_decode_motion(s, s->mpeg_f_code[i][1],

                                                 s->last_mv[i][0][1] >> my_shift);

                        dmy = get_dmv(s);





                        s->last_mv[i][0][1] = my << my_shift;

                        s->last_mv[i][1][1] = my << my_shift;



                        s->mv[i][0][0] = mx;

                        s->mv[i][0][1] = my;

                        s->mv[i][1][0] = mx; // not used

                        s->mv[i][1][1] = my; // not used



                        if (s->picture_structure == PICT_FRAME) {

                            mb_type |= MB_TYPE_16x16 | MB_TYPE_INTERLACED;



                            // m = 1 + 2 * s->top_field_first;

                            m = s->top_field_first ? 1 : 3;



                            /* top -> top pred */

                            s->mv[i][2][0] = ((mx * m + (mx > 0)) >> 1) + dmx;

                            s->mv[i][2][1] = ((my * m + (my > 0)) >> 1) + dmy - 1;

                            m = 4 - m;

                            s->mv[i][3][0] = ((mx * m + (mx > 0)) >> 1) + dmx;

                            s->mv[i][3][1] = ((my * m + (my > 0)) >> 1) + dmy + 1;

                        } else {

                            mb_type |= MB_TYPE_16x16;



                            s->mv[i][2][0] = ((mx + (mx > 0)) >> 1) + dmx;

                            s->mv[i][2][1] = ((my + (my > 0)) >> 1) + dmy;

                            if (s->picture_structure == PICT_TOP_FIELD)

                                s->mv[i][2][1]--;

                            else

                                s->mv[i][2][1]++;

                        }

                    }

                }

                break;

            default:

                av_log(s->avctx, AV_LOG_ERROR, "00 motion_type at %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }

        }



        s->mb_intra = 0;

        if (HAS_CBP(mb_type)) {

            s->dsp.clear_blocks(s->block[0]);



            cbp = get_vlc2(&s->gb, mb_pat_vlc.table, MB_PAT_VLC_BITS, 1);

            if (mb_block_count > 6) {

                 cbp <<= mb_block_count - 6;

                 cbp  |= get_bits(&s->gb, mb_block_count - 6);

                 s->dsp.clear_blocks(s->block[6]);

            }

            if (cbp <= 0) {

                av_log(s->avctx, AV_LOG_ERROR, "invalid cbp at %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }



            //if 1, we memcpy blocks in xvmcvideo

            if (CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration > 1) {

                ff_xvmc_pack_pblocks(s, cbp);

                if (s->swap_uv) {

                    exchange_uv(s);

                }

            }



            if (s->codec_id == CODEC_ID_MPEG2VIDEO) {

                if (s->flags2 & CODEC_FLAG2_FAST) {

                    for (i = 0; i < 6; i++) {

                        if (cbp & 32) {

                            mpeg2_fast_decode_block_non_intra(s, *s->pblocks[i], i);

                        } else {

                            s->block_last_index[i] = -1;

                        }

                        cbp += cbp;

                    }

                } else {

                    cbp <<= 12-mb_block_count;



                    for (i = 0; i < mb_block_count; i++) {

                        if (cbp & (1 << 11)) {

                            if (mpeg2_decode_block_non_intra(s, *s->pblocks[i], i) < 0)

                                return -1;

                        } else {

                            s->block_last_index[i] = -1;

                        }

                        cbp += cbp;

                    }

                }

            } else {

                if (s->flags2 & CODEC_FLAG2_FAST) {

                    for (i = 0; i < 6; i++) {

                        if (cbp & 32) {

                            mpeg1_fast_decode_block_inter(s, *s->pblocks[i], i);

                        } else {

                            s->block_last_index[i] = -1;

                        }

                        cbp += cbp;

                    }

                } else {

                    for (i = 0; i < 6; i++) {

                        if (cbp & 32) {

                            if (mpeg1_decode_block_inter(s, *s->pblocks[i], i) < 0)

                                return -1;

                        } else {

                            s->block_last_index[i] = -1;

                        }

                        cbp += cbp;

                    }

                }

            }

        } else {

            for (i = 0; i < 12; i++)

                s->block_last_index[i] = -1;

        }

    }



    s->current_picture.f.mb_type[s->mb_x + s->mb_y * s->mb_stride] = mb_type;



    return 0;

}
