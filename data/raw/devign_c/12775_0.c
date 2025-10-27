static void unpack_modes(Vp3DecodeContext *s, GetBitContext *gb)

{

    int i, j, k;

    int scheme;

    int current_macroblock;

    int current_fragment;

    int coding_mode;



    debug_vp3("  vp3: unpacking encoding modes\n");



    if (s->keyframe) {

        debug_vp3("    keyframe-- all blocks are coded as INTRA\n");



        for (i = 0; i < s->fragment_count; i++)

            s->all_fragments[i].coding_method = MODE_INTRA;



    } else {



        /* fetch the mode coding scheme for this frame */

        scheme = get_bits(gb, 3);

        debug_modes("    using mode alphabet %d\n", scheme);



        /* is it a custom coding scheme? */

        if (scheme == 0) {

            debug_modes("    custom mode alphabet ahead:\n");

            for (i = 0; i < 8; i++)

                ModeAlphabet[0][i] = get_bits(gb, 3);

        }



        for (i = 0; i < 8; i++)

            debug_modes("      mode[%d][%d] = %d\n", scheme, i, 

                ModeAlphabet[scheme][i]);



        /* iterate through all of the macroblocks that contain 1 or more

         * coded fragments */

        for (i = 0; i < s->u_superblock_start; i++) {



            for (j = 0; j < 4; j++) {

                current_macroblock = s->superblock_macroblocks[i * 4 + j];

                if ((current_macroblock == -1) ||

                    (!s->macroblock_coded[current_macroblock]))

                    continue;



                /* mode 7 means get 3 bits for each coding mode */

                if (scheme == 7)

                    coding_mode = get_bits(gb, 3);

                else

                    coding_mode = ModeAlphabet[scheme][get_mode_code(gb)];



                for (k = 0; k < 6; k++) {

                    current_fragment = 

                        s->macroblock_fragments[current_macroblock * 6 + k];

                    if (s->all_fragments[current_fragment].coding_method != 

                        MODE_COPY)

                        s->all_fragments[current_fragment].coding_method =

                            coding_mode;

                }



                debug_modes("    coding method for macroblock starting @ fragment %d = %d\n",

                    s->macroblock_fragments[current_macroblock * 6], coding_mode);

            }

        }

    }

}
