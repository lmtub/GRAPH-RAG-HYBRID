static inline int mpeg1_fast_decode_block_inter(MpegEncContext *s, int16_t *block, int n)

{

    int level, i, j, run;

    RLTable *rl = &ff_rl_mpeg1;

    uint8_t * const scantable = s->intra_scantable.permutated;

    const int qscale          = s->qscale;



    {

        OPEN_READER(re, &s->gb);

        i = -1;

        // special case for first coefficient, no need to add second VLC table

        UPDATE_CACHE(re, &s->gb);

        if (((int32_t)GET_CACHE(re, &s->gb)) < 0) {

            level = (3 * qscale) >> 1;

            level = (level - 1) | 1;

            if (GET_CACHE(re, &s->gb) & 0x40000000)

                level = -level;

            block[0] = level;

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

                level = ((level * 2 + 1) * qscale) >> 1;

                level = (level - 1) | 1;

                level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

                SKIP_BITS(re, &s->gb, 1);

            } else {

                /* escape */

                run = SHOW_UBITS(re, &s->gb, 6)+1; LAST_SKIP_BITS(re, &s->gb, 6);

                UPDATE_CACHE(re, &s->gb);

                level = SHOW_SBITS(re, &s->gb, 8); SKIP_BITS(re, &s->gb, 8);

                if (level == -128) {

                    level = SHOW_UBITS(re, &s->gb, 8) - 256; SKIP_BITS(re, &s->gb, 8);

                } else if (level == 0) {

                    level = SHOW_UBITS(re, &s->gb, 8)      ; SKIP_BITS(re, &s->gb, 8);

                }

                i += run;

                j = scantable[i];

                if (level < 0) {

                    level = -level;

                    level = ((level * 2 + 1) * qscale) >> 1;

                    level = (level - 1) | 1;

                    level = -level;

                } else {

                    level = ((level * 2 + 1) * qscale) >> 1;

                    level = (level - 1) | 1;

                }

            }



            block[j] = level;

            if (((int32_t)GET_CACHE(re, &s->gb)) <= (int32_t)0xBFFFFFFF)

                break;

            UPDATE_CACHE(re, &s->gb);

        }

end:

        LAST_SKIP_BITS(re, &s->gb, 2);

        CLOSE_READER(re, &s->gb);

    }

    s->block_last_index[n] = i;

    return 0;

}
