static int h263_decode_block(MpegEncContext * s, int16_t * block,

                             int n, int coded)

{

    int level, i, j, run;

    RLTable *rl = &ff_h263_rl_inter;

    const uint8_t *scan_table;

    GetBitContext gb= s->gb;



    scan_table = s->intra_scantable.permutated;

    if (s->h263_aic && s->mb_intra) {

        rl = &ff_rl_intra_aic;

        i = 0;

        if (s->ac_pred) {

            if (s->h263_aic_dir)

                scan_table = s->intra_v_scantable.permutated; /* left */

            else

                scan_table = s->intra_h_scantable.permutated; /* top */

        }

    } else if (s->mb_intra) {

        /* DC coef */

        if (CONFIG_RV10_DECODER && s->codec_id == AV_CODEC_ID_RV10) {

          if (s->rv10_version == 3 && s->pict_type == AV_PICTURE_TYPE_I) {

            int component, diff;

            component = (n <= 3 ? 0 : n - 4 + 1);

            level = s->last_dc[component];

            if (s->rv10_first_dc_coded[component]) {

                diff = ff_rv_decode_dc(s, n);

                if (diff == 0xffff)

                    return -1;

                level += diff;

                level = level & 0xff; /* handle wrap round */

                s->last_dc[component] = level;

            } else {

                s->rv10_first_dc_coded[component] = 1;

            }

          } else {

                level = get_bits(&s->gb, 8);

                if (level == 255)

                    level = 128;

          }

        }else{

            level = get_bits(&s->gb, 8);

            if((level&0x7F) == 0){

                av_log(s->avctx, AV_LOG_ERROR, "illegal dc %d at %d %d\n", level, s->mb_x, s->mb_y);

                if (s->avctx->err_recognition & (AV_EF_BITSTREAM|AV_EF_COMPLIANT))

                    return -1;

            }

            if (level == 255)

                level = 128;

        }

        block[0] = level;

        i = 1;

    } else {

        i = 0;

    }

    if (!coded) {

        if (s->mb_intra && s->h263_aic)

            goto not_coded;

        s->block_last_index[n] = i - 1;

        return 0;

    }

retry:

    {

    OPEN_READER(re, &s->gb);

    i--; // offset by -1 to allow direct indexing of scan_table

    for(;;) {

        UPDATE_CACHE(re, &s->gb);

        GET_RL_VLC(level, run, re, &s->gb, rl->rl_vlc[0], TEX_VLC_BITS, 2, 0);

        if (run == 66) {

            if (level){

                CLOSE_READER(re, &s->gb);

                av_log(s->avctx, AV_LOG_ERROR, "illegal ac vlc code at %dx%d\n", s->mb_x, s->mb_y);

                return -1;

            }

            /* escape */

            if (CONFIG_FLV_DECODER && s->h263_flv > 1) {

                int is11 = SHOW_UBITS(re, &s->gb, 1);

                SKIP_CACHE(re, &s->gb, 1);

                run = SHOW_UBITS(re, &s->gb, 7) + 1;

                if (is11) {

                    SKIP_COUNTER(re, &s->gb, 1 + 7);

                    UPDATE_CACHE(re, &s->gb);

                    level = SHOW_SBITS(re, &s->gb, 11);

                    SKIP_COUNTER(re, &s->gb, 11);

                } else {

                    SKIP_CACHE(re, &s->gb, 7);

                    level = SHOW_SBITS(re, &s->gb, 7);

                    SKIP_COUNTER(re, &s->gb, 1 + 7 + 7);

                }

            } else {

                run = SHOW_UBITS(re, &s->gb, 7) + 1;

                SKIP_CACHE(re, &s->gb, 7);

                level = (int8_t)SHOW_UBITS(re, &s->gb, 8);

                SKIP_COUNTER(re, &s->gb, 7 + 8);

                if(level == -128){

                    UPDATE_CACHE(re, &s->gb);

                    if (s->codec_id == AV_CODEC_ID_RV10) {

                        /* XXX: should patch encoder too */

                        level = SHOW_SBITS(re, &s->gb, 12);

                        SKIP_COUNTER(re, &s->gb, 12);

                    }else{

                        level = SHOW_UBITS(re, &s->gb, 5);

                        SKIP_CACHE(re, &s->gb, 5);

                        level |= SHOW_SBITS(re, &s->gb, 6)<<5;

                        SKIP_COUNTER(re, &s->gb, 5 + 6);

                    }

                }

            }

        } else {

            if (SHOW_UBITS(re, &s->gb, 1))

                level = -level;

            SKIP_COUNTER(re, &s->gb, 1);

        }

        i += run;

        if (i >= 64){

            CLOSE_READER(re, &s->gb);

            // redo update without last flag, revert -1 offset

            i = i - run + ((run-1)&63) + 1;

            if (i < 64) {

                // only last marker, no overrun

                block[scan_table[i]] = level;

                break;

            }

            if(s->alt_inter_vlc && rl == &ff_h263_rl_inter && !s->mb_intra){

                //Looks like a hack but no, it's the way it is supposed to work ...

                rl = &ff_rl_intra_aic;

                i = 0;

                s->gb= gb;

                s->bdsp.clear_block(block);

                goto retry;

            }

            av_log(s->avctx, AV_LOG_ERROR, "run overflow at %dx%d i:%d\n", s->mb_x, s->mb_y, s->mb_intra);

            return -1;

        }

        j = scan_table[i];

        block[j] = level;

    }

    }

not_coded:

    if (s->mb_intra && s->h263_aic) {

        ff_h263_pred_acdc(s, block, n);

        i = 63;

    }

    s->block_last_index[n] = i;

    return 0;

}
