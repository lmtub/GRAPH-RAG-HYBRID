static int unpack_superblocks(Vp3DecodeContext *s, GetBitContext *gb)

{

    int superblock_starts[3] = { 0, s->u_superblock_start, s->v_superblock_start };

    int bit = 0;

    int current_superblock = 0;

    int current_run = 0;

    int num_partial_superblocks = 0;



    int i, j;

    int current_fragment;

    int plane;



    if (s->keyframe) {

        memset(s->superblock_coding, SB_FULLY_CODED, s->superblock_count);



    } else {



        /* unpack the list of partially-coded superblocks */

        bit = get_bits1(gb);

        while (current_superblock < s->superblock_count) {

                current_run = get_vlc2(gb,

                    s->superblock_run_length_vlc.table, 6, 2) + 1;

                if (current_run == 34)

                    current_run += get_bits(gb, 12);



            if (current_superblock + current_run > s->superblock_count) {

                av_log(s->avctx, AV_LOG_ERROR, "Invalid partially coded superblock run length\n");

                return -1;

            }



            memset(s->superblock_coding + current_superblock, bit, current_run);



            current_superblock += current_run;

            if (bit)

                num_partial_superblocks += current_run;



            if (s->theora && current_run == MAXIMUM_LONG_BIT_RUN)

                bit = get_bits1(gb);

            else

                bit ^= 1;

        }



        /* unpack the list of fully coded superblocks if any of the blocks were

         * not marked as partially coded in the previous step */

        if (num_partial_superblocks < s->superblock_count) {

            int superblocks_decoded = 0;



            current_superblock = 0;

            bit = get_bits1(gb);

            while (superblocks_decoded < s->superblock_count - num_partial_superblocks) {

                        current_run = get_vlc2(gb,

                            s->superblock_run_length_vlc.table, 6, 2) + 1;

                        if (current_run == 34)

                            current_run += get_bits(gb, 12);



                for (j = 0; j < current_run; current_superblock++) {

                    if (current_superblock >= s->superblock_count) {

                        av_log(s->avctx, AV_LOG_ERROR, "Invalid fully coded superblock run length\n");

                        return -1;

                    }



                /* skip any superblocks already marked as partially coded */

                if (s->superblock_coding[current_superblock] == SB_NOT_CODED) {

                    s->superblock_coding[current_superblock] = 2*bit;

                    j++;

                }

                }

                superblocks_decoded += current_run;



                if (s->theora && current_run == MAXIMUM_LONG_BIT_RUN)

                    bit = get_bits1(gb);

                else

                    bit ^= 1;

            }

        }



        /* if there were partial blocks, initialize bitstream for

         * unpacking fragment codings */

        if (num_partial_superblocks) {



            current_run = 0;

            bit = get_bits1(gb);

            /* toggle the bit because as soon as the first run length is

             * fetched the bit will be toggled again */

            bit ^= 1;

        }

    }



    /* figure out which fragments are coded; iterate through each

     * superblock (all planes) */

    s->total_num_coded_frags = 0;

    memset(s->macroblock_coding, MODE_COPY, s->macroblock_count);



    for (plane = 0; plane < 3; plane++) {

        int sb_start = superblock_starts[plane];

        int sb_end = sb_start + (plane ? s->c_superblock_count : s->y_superblock_count);

        int num_coded_frags = 0;



    for (i = sb_start; i < sb_end; i++) {



        /* iterate through all 16 fragments in a superblock */

        for (j = 0; j < 16; j++) {



            /* if the fragment is in bounds, check its coding status */

            current_fragment = s->superblock_fragments[i * 16 + j];

            if (current_fragment != -1) {

                int coded = s->superblock_coding[i];



                if (s->superblock_coding[i] == SB_PARTIALLY_CODED) {



                    /* fragment may or may not be coded; this is the case

                     * that cares about the fragment coding runs */

                    if (current_run-- == 0) {

                        bit ^= 1;

                        current_run = get_vlc2(gb,

                            s->fragment_run_length_vlc.table, 5, 2);

                    }

                    coded = bit;

                }



                    if (coded) {

                        /* default mode; actual mode will be decoded in

                         * the next phase */

                        s->all_fragments[current_fragment].coding_method =

                            MODE_INTER_NO_MV;

                        s->coded_fragment_list[plane][num_coded_frags++] =

                            current_fragment;

                    } else {

                        /* not coded; copy this fragment from the prior frame */

                        s->all_fragments[current_fragment].coding_method =

                            MODE_COPY;

                    }

            }

        }

    }

        s->total_num_coded_frags += num_coded_frags;

        for (i = 0; i < 64; i++)

            s->num_coded_frags[plane][i] = num_coded_frags;

        if (plane < 2)

            s->coded_fragment_list[plane+1] = s->coded_fragment_list[plane] + num_coded_frags;

    }

    return 0;

}
