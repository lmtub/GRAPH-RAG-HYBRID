void ff_eac3_output_frame_header(AC3EncodeContext *s)

{

    int blk, ch;

    AC3EncOptions *opt = &s->options;



    put_bits(&s->pb, 16, 0x0b77);                   /* sync word */



    /* BSI header */

    put_bits(&s->pb,  2, 0);                        /* stream type = independent */

    put_bits(&s->pb,  3, 0);                        /* substream id = 0 */

    put_bits(&s->pb, 11, (s->frame_size / 2) - 1);  /* frame size */

    if (s->bit_alloc.sr_shift) {

        put_bits(&s->pb, 2, 0x3);                   /* fscod2 */

        put_bits(&s->pb, 2, s->bit_alloc.sr_code);  /* sample rate code */

    } else {

        put_bits(&s->pb, 2, s->bit_alloc.sr_code);  /* sample rate code */

        put_bits(&s->pb, 2, 0x3);                   /* number of blocks = 6 */

    }

    put_bits(&s->pb, 3, s->channel_mode);           /* audio coding mode */

    put_bits(&s->pb, 1, s->lfe_on);                 /* LFE channel indicator */

    put_bits(&s->pb, 5, s->bitstream_id);           /* bitstream id (EAC3=16) */

    put_bits(&s->pb, 5, -opt->dialogue_level);      /* dialogue normalization level */

    put_bits(&s->pb, 1, 0);                         /* no compression gain */

    put_bits(&s->pb, 1, 0);                         /* no mixing metadata */

    /* TODO: mixing metadata */

    put_bits(&s->pb, 1, 0);                         /* no info metadata */

    /* TODO: info metadata */

    put_bits(&s->pb, 1, 0);                         /* no additional bit stream info */



    /* frame header */

    put_bits(&s->pb, 1, 1);                         /* exponent strategy syntax = each block */

    put_bits(&s->pb, 1, 0);                         /* aht enabled = no */

    put_bits(&s->pb, 2, 0);                         /* snr offset strategy = 1 */

    put_bits(&s->pb, 1, 0);                         /* transient pre-noise processing enabled = no */

    put_bits(&s->pb, 1, 0);                         /* block switch syntax enabled = no */

    put_bits(&s->pb, 1, 0);                         /* dither flag syntax enabled = no */

    put_bits(&s->pb, 1, 0);                         /* bit allocation model syntax enabled = no */

    put_bits(&s->pb, 1, 0);                         /* fast gain codes enabled = no */

    put_bits(&s->pb, 1, 0);                         /* dba syntax enabled = no */

    put_bits(&s->pb, 1, 0);                         /* skip field syntax enabled = no */

    put_bits(&s->pb, 1, 0);                         /* spx enabled = no */

    /* coupling strategy use flags */

    if (s->channel_mode > AC3_CHMODE_MONO) {

        put_bits(&s->pb, 1, s->blocks[0].cpl_in_use);

        for (blk = 1; blk < AC3_MAX_BLOCKS; blk++) {

            AC3Block *block = &s->blocks[blk];

            put_bits(&s->pb, 1, block->new_cpl_strategy);

            if (block->new_cpl_strategy)

                put_bits(&s->pb, 1, block->cpl_in_use);

        }

    }

    /* exponent strategy */

    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++)

        for (ch = !s->blocks[blk].cpl_in_use; ch <= s->fbw_channels; ch++)

            put_bits(&s->pb, 2, s->exp_strategy[ch][blk]);

    if (s->lfe_on) {

        for (blk = 0; blk < AC3_MAX_BLOCKS; blk++)

            put_bits(&s->pb, 1, s->exp_strategy[s->lfe_channel][blk]);

    }

    /* E-AC-3 to AC-3 converter exponent strategy (unfortunately not optional...) */

    for (ch = 1; ch <= s->fbw_channels; ch++)

        put_bits(&s->pb, 5, 0);

    /* snr offsets */

    put_bits(&s->pb, 6, s->coarse_snr_offset);

    put_bits(&s->pb, 4, s->fine_snr_offset[1]);

    /* block start info */

    put_bits(&s->pb, 1, 0);

}
