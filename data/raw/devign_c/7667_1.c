static inline int mpeg2_decode_block_non_intra(MpegEncContext *s, int16_t *block, int n)

{

    int level, i, j, run;

    RLTable *rl = &ff_rl_mpeg1;

    uint8_t * const scantable = s->intra_scantable.permutated;

    const uint16_t *quant_matrix;

    const int qscale = s->qscale;

    int mismatch;



    mismatch = 1;



    {

        OPEN_READER(re, &s->gb);

        i = -1;

        if (n < 4)

            quant_matrix = s->inter_matrix;

        else

            quant_matrix = s->chroma_inter_matrix;



        // special case for first coefficient, no need to add second VLC table

        UPDATE_CACHE(re, &s->gb);

        if (((int32_t)GET_CACHE(re, &s->gb)) < 0) {

            level= (3 * qscale * quant_matrix[0]) >> 5;

            if (GET_CACHE(re, &s->gb) & 0x40000000)

                level = -level;

            block[0]  = level;

            mismatch ^= level;

            i++;

            SKIP_BITS(re, &s->gb, 2);

            if (((int32_t)GET_CACHE(re, &s->gb)) <= (int32_t)0xBFFFFFFF)

                goto end;

        }



        /* now quantify & encode AC coefficients */

        for (;;) {

            GET_RL_VLC(level, run, re, &s->gb, rl->rl_vlc[0], TEX_VLC_BITS, 2, 0);



            if (level != 0) {

                i += run;

                j = scantable[i];

                level = ((level * 2 + 1) * qscale * quant_matrix[j]) >> 5;

                level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

                SKIP_BITS(re, &s->gb, 1);

            } else {

                /* escape */

                run = SHOW_UBITS(re, &s->gb, 6) + 1; LAST_SKIP_BITS(re, &s->gb, 6);

                UPDATE_CACHE(re, &s->gb);

                level = SHOW_SBITS(re, &s->gb, 12); SKIP_BITS(re, &s->gb, 12);



                i += run;

                j = scantable[i];

                if (level < 0) {

                    level = ((-level * 2 + 1) * qscale * quant_matrix[j]) >> 5;

                    level = -level;

                } else {

                    level = ((level * 2 + 1) * qscale * quant_matrix[j]) >> 5;

                }

            }

            if (i > 63) {

                av_log(s->avctx, AV_LOG_ERROR, "ac-tex damaged at %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }



            mismatch ^= level;

            block[j]  = level;

            if (((int32_t)GET_CACHE(re, &s->gb)) <= (int32_t)0xBFFFFFFF)

                break;

            UPDATE_CACHE(re, &s->gb);

        }

end:

        LAST_SKIP_BITS(re, &s->gb, 2);

        CLOSE_READER(re, &s->gb);

    }

    block[63] ^= (mismatch & 1);



    s->block_last_index[n] = i;

    return 0;

}
