static void count_frame_bits_fixed(AC3EncodeContext *s)

{

    static const int frame_bits_inc[8] = { 0, 0, 2, 2, 2, 4, 2, 4 };

    int blk;

    int frame_bits;



    /* assumptions:

     *   no dynamic range codes

     *   bit allocation parameters do not change between blocks

     *   no delta bit allocation

     *   no skipped data

     *   no auxilliary data

     *   no E-AC-3 metadata

     */



    /* header */

    frame_bits = 16; /* sync info */

    if (s->eac3) {

        /* bitstream info header */

        frame_bits += 35;

        frame_bits += 1 + 1 + 1;

        /* audio frame header */

        frame_bits += 2;

        frame_bits += 10;

        /* exponent strategy */

        for (blk = 0; blk < AC3_MAX_BLOCKS; blk++)

            frame_bits += 2 * s->fbw_channels + s->lfe_on;

        /* converter exponent strategy */

        frame_bits += s->fbw_channels * 5;

        /* snr offsets */

        frame_bits += 10;

        /* block start info */

        frame_bits++;

    } else {

        frame_bits += 49;

        frame_bits += frame_bits_inc[s->channel_mode];

    }



    /* audio blocks */

    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {

        if (!s->eac3) {

            /* block switch flags */

            frame_bits += s->fbw_channels;



            /* dither flags */

            frame_bits += s->fbw_channels;

        }



        /* dynamic range */

        frame_bits++;



        /* spectral extension */

        if (s->eac3)

            frame_bits++;



        if (!s->eac3) {

            /* exponent strategy */

            frame_bits += 2 * s->fbw_channels;

            if (s->lfe_on)

                frame_bits++;



            /* bit allocation params */

            frame_bits++;

            if (!blk)

                frame_bits += 2 + 2 + 2 + 2 + 3;

        }



        /* converter snr offset */

        if (s->eac3)

            frame_bits++;



        if (!s->eac3) {

            /* delta bit allocation */

            frame_bits++;



            /* skipped data */

            frame_bits++;

        }

    }



    /* auxiliary data */

    frame_bits++;



    /* CRC */

    frame_bits += 1 + 16;



    s->frame_bits_fixed = frame_bits;

}
