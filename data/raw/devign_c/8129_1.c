static int unpack_modes(Vp3DecodeContext *s, GetBitContext *gb)

{

    int i, j, k;

    int scheme;

    int current_macroblock;

    int current_fragment;

    int coding_mode;

    int custom_mode_alphabet[CODING_MODE_COUNT];



    if (s->keyframe) {

        for (i = 0; i < s->fragment_count; i++)

            s->all_fragments[i].coding_method = MODE_INTRA;



    } else {



        /* fetch the mode coding scheme for this frame */

        scheme = get_bits(gb, 3);



        /* is it a custom coding scheme? */

        if (scheme == 0) {




                custom_mode_alphabet[get_bits(gb, 3)] = i;

        }



        /* iterate through all of the macroblocks that contain 1 or more

         * coded fragments */

        for (i = 0; i < s->u_superblock_start; i++) {



            for (j = 0; j < 4; j++) {

                current_macroblock = s->superblock_macroblocks[i * 4 + j];

                if ((current_macroblock == -1) ||

                    (s->macroblock_coding[current_macroblock] == MODE_COPY))

                    continue;

                if (current_macroblock >= s->macroblock_count) {

                    av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_modes(): bad macroblock number (%d >= %d)\n",

                        current_macroblock, s->macroblock_count);

                    return 1;

                }



                /* mode 7 means get 3 bits for each coding mode */

                if (scheme == 7)

                    coding_mode = get_bits(gb, 3);

                else if(scheme == 0)

                    coding_mode = custom_mode_alphabet

                        [get_vlc2(gb, s->mode_code_vlc.table, 3, 3)];

                else

                    coding_mode = ModeAlphabet[scheme-1]

                        [get_vlc2(gb, s->mode_code_vlc.table, 3, 3)];



                s->macroblock_coding[current_macroblock] = coding_mode;

                for (k = 0; k < 6; k++) {

                    current_fragment =

                        s->macroblock_fragments[current_macroblock * 6 + k];

                    if (current_fragment == -1)

                        continue;

                    if (current_fragment >= s->fragment_count) {

                        av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_modes(): bad fragment number (%d >= %d)\n",

                            current_fragment, s->fragment_count);

                        return 1;

                    }

                    if (s->all_fragments[current_fragment].coding_method !=

                        MODE_COPY)

                        s->all_fragments[current_fragment].coding_method =

                            coding_mode;

                }

            }

        }

    }



    return 0;

}