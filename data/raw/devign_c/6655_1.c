static int mpeg2_decode_block_non_intra(MpegEncContext *s, 

                                        DCTELEM *block, 

                                        int n)

{

    int level, i, j, run;

    int code;

    RLTable *rl = &rl_mpeg1;

    const UINT8 *scan_table;

    const UINT16 *matrix;

    int mismatch;



    if (s->alternate_scan)

        scan_table = ff_alternate_vertical_scan;

    else

        scan_table = zigzag_direct;

    mismatch = 1;



    {

        int bit_cnt, v;

        UINT32 bit_buf;

        UINT8 *buf_ptr;

        i = 0;

        if (n < 4) 

            matrix = s->non_intra_matrix;

        else

            matrix = s->chroma_non_intra_matrix;

            

        /* special case for the first coef. no need to add a second vlc table */

        SAVE_BITS(&s->gb);

        SHOW_BITS(&s->gb, v, 2);

        if (v & 2) {

            run = 0;

            level = 1 - ((v & 1) << 1);

            FLUSH_BITS(2);

            RESTORE_BITS(&s->gb);

            goto add_coef;

        }

        RESTORE_BITS(&s->gb);

    }



    /* now quantify & encode AC coefs */

    for(;;) {

        code = get_vlc(&s->gb, &rl->vlc);

        if (code < 0)

            return -1;

        if (code == 112) {

            break;

        } else if (code == 111) {

            /* escape */

            run = get_bits(&s->gb, 6);

            level = get_bits(&s->gb, 12);

            level = (level << 20) >> 20;

        } else {

            run = rl->table_run[code];

            level = rl->table_level[code];

            if (get_bits1(&s->gb))

                level = -level;

        }

        i += run;

        if (i >= 64)

            return -1;

    add_coef:

	j = scan_table[i];

        dprintf("%d: run=%d level=%d\n", n, run, level);

        /* XXX: optimize */

        if (level > 0) {

            level = ((level * 2 + 1) * s->qscale * matrix[j]) >> 5;

        } else {

            level = ((-level * 2 + 1) * s->qscale * matrix[j]) >> 5;

            level = -level;

        }

        /* XXX: is it really necessary to saturate since the encoder

           knows whats going on ? */

        mismatch ^= level;

        block[j] = level;

        i++;

    }

    block[63] ^= (mismatch & 1);

    s->block_last_index[n] = i;

    return 0;

}
