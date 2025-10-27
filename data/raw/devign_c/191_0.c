static void dv_decode_ac(DVVideoDecodeContext *s, 

                         BlockInfo *mb, DCTELEM *block, int last_index)

{

    int last_re_index;

    int shift_offset = mb->shift_offset;

    const UINT8 *scan_table = mb->scan_table;

    const UINT8 *shift_table = mb->shift_table;

    int pos = mb->pos;

    int level, pos1, sign, run;

    int partial_bit_count;



    OPEN_READER(re, &s->gb);

    

#ifdef VLC_DEBUG

    printf("start\n");

#endif



    /* if we must parse a partial vlc, we do it here */

    partial_bit_count = mb->partial_bit_count;

    if (partial_bit_count > 0) {

        UINT8 buf[4];

        UINT32 v;

        int l, l1;

        GetBitContext gb1;



        /* build the dummy bit buffer */

        l = 16 - partial_bit_count;

        UPDATE_CACHE(re, &s->gb);

#ifdef VLC_DEBUG

        printf("show=%04x\n", SHOW_UBITS(re, &s->gb, 16));

#endif

        v = (mb->partial_bit_buffer << l) | SHOW_UBITS(re, &s->gb, l);

        buf[0] = v >> 8;

        buf[1] = v;

#ifdef VLC_DEBUG

        printf("v=%04x cnt=%d %04x\n", 

               v, partial_bit_count, (mb->partial_bit_buffer << l));

#endif

        /* try to read the codeword */

        init_get_bits(&gb1, buf, 4);

        {

            OPEN_READER(re1, &gb1);

            UPDATE_CACHE(re1, &gb1);

            GET_RL_VLC(level, run, re1, &gb1, dv_rl_vlc[0], 

                       TEX_VLC_BITS, 2);

            l = re1_index;

            CLOSE_READER(re1, &gb1);

        }

#ifdef VLC_DEBUG

        printf("****run=%d level=%d size=%d\n", run, level, l);

#endif

        /* compute codeword length */

        l1 = (level != 256 && level != 0);

        /* if too long, we cannot parse */

        l -= partial_bit_count;

        if ((re_index + l + l1) > last_index)

            return;

        /* skip read bits */

        last_re_index = 0; /* avoid warning */

        re_index += l;

        /* by definition, if we can read the vlc, all partial bits

           will be read (otherwise we could have read the vlc before) */

        mb->partial_bit_count = 0;

        UPDATE_CACHE(re, &s->gb);

        goto handle_vlc;

    }



    /* get the AC coefficients until last_index is reached */

    for(;;) {

        UPDATE_CACHE(re, &s->gb);

#ifdef VLC_DEBUG

        printf("%2d: bits=%04x index=%d\n", 

               pos, SHOW_UBITS(re, &s->gb, 16), re_index);

#endif

        last_re_index = re_index;

        GET_RL_VLC(level, run, re, &s->gb, dv_rl_vlc[0], 

                   TEX_VLC_BITS, 2);

    handle_vlc:

#ifdef VLC_DEBUG

        printf("run=%d level=%d\n", run, level);

#endif

        if (level == 256) {

            if (re_index > last_index) {

            cannot_read:

                /* put position before read code */

                re_index = last_re_index;

                mb->eob_reached = 0;

                break;

            }

            /* EOB */

            mb->eob_reached = 1;

            break;

        } else if (level != 0) {

            if ((re_index + 1) > last_index)

                goto cannot_read;

            sign = SHOW_SBITS(re, &s->gb, 1);

            level = (level ^ sign) - sign;

            LAST_SKIP_BITS(re, &s->gb, 1);

            pos += run;

            /* error */

            if (pos >= 64) {

                goto read_error;

            }

            pos1 = scan_table[pos];

            level = level << (shift_table[pos1] + shift_offset);

            block[pos1] = level;

            //            printf("run=%d level=%d shift=%d\n", run, level, shift_table[pos1]);

        } else {

            if (re_index > last_index)

                goto cannot_read;

            /* level is zero: means run without coding. No

               sign is coded */

            pos += run;

            /* error */

            if (pos >= 64) {

            read_error:

#if defined(VLC_DEBUG) || 1

                printf("error pos=%d\n", pos);

#endif

                /* for errors, we consider the eob is reached */

                mb->eob_reached = 1;

                break;

            }

        }

    }

    CLOSE_READER(re, &s->gb);

    mb->pos = pos;

}
