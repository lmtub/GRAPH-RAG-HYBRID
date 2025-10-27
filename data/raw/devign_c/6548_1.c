static inline int mpeg1_decode_block_intra(MpegEncContext *s, int16_t *block, int n)

{

    int level, dc, diff, i, j, run;

    int component;

    RLTable *rl = &ff_rl_mpeg1;

    uint8_t * const scantable    = s->intra_scantable.permutated;

    const uint16_t *quant_matrix = s->intra_matrix;

    const int qscale             = s->qscale;



    /* DC coefficient */

    component = (n <= 3 ? 0 : n - 4 + 1);

    diff = decode_dc(&s->gb, component);

    if (diff >= 0xffff)

        return -1;

    dc  = s->last_dc[component];

    dc += diff;

    s->last_dc[component] = dc;

    block[0] = dc * quant_matrix[0];

    av_dlog(s->avctx, "dc=%d diff=%d\n", dc, diff);

    i = 0;

    {

        OPEN_READER(re, &s->gb);

        /* now quantify & encode AC coefficients */

        for (;;) {

            UPDATE_CACHE(re, &s->gb);

            GET_RL_VLC(level, run, re, &s->gb, rl->rl_vlc[0], TEX_VLC_BITS, 2, 0);



            if (level == 127) {

                break;

            } else if (level != 0) {

                i += run;

                j = scantable[i];

                level = (level * qscale * quant_matrix[j]) >> 4;

                level = (level - 1) | 1;

                level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

                LAST_SKIP_BITS(re, &s->gb, 1);

            } else {

                /* escape */

                run = SHOW_UBITS(re, &s->gb, 6) + 1; LAST_SKIP_BITS(re, &s->gb, 6);

                UPDATE_CACHE(re, &s->gb);

                level = SHOW_SBITS(re, &s->gb, 8); SKIP_BITS(re, &s->gb, 8);

                if (level == -128) {

                    level = SHOW_UBITS(re, &s->gb, 8) - 256; LAST_SKIP_BITS(re, &s->gb, 8);

                } else if (level == 0) {

                    level = SHOW_UBITS(re, &s->gb, 8)      ; LAST_SKIP_BITS(re, &s->gb, 8);

                }

                i += run;

                j = scantable[i];

                if (level < 0) {

                    level = -level;

                    level = (level * qscale * quant_matrix[j]) >> 4;

                    level = (level - 1) | 1;

                    level = -level;

                } else {

                    level = (level * qscale * quant_matrix[j]) >> 4;

                    level = (level - 1) | 1;

                }

            }

            if (i > 63) {

                av_log(s->avctx, AV_LOG_ERROR, "ac-tex damaged at %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }



            block[j] = level;

        }

        CLOSE_READER(re, &s->gb);

    }

    s->block_last_index[n] = i;

   return 0;

}
