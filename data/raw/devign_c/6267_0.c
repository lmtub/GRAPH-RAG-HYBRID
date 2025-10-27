static int unpack_vectors(Vp3DecodeContext *s, GetBitContext *gb)

{

    int j, k, l, sb_x, sb_y;

    int coding_mode;

    int motion_x[6];

    int motion_y[6];

    int last_motion_x = 0;

    int last_motion_y = 0;

    int prior_last_motion_x = 0;

    int prior_last_motion_y = 0;

    int current_macroblock;

    int current_fragment;



    if (s->keyframe)

        return 0;



    memset(motion_x, 0, 6 * sizeof(int));

    memset(motion_y, 0, 6 * sizeof(int));



    /* coding mode 0 is the VLC scheme; 1 is the fixed code scheme */

    coding_mode = get_bits1(gb);



    /* iterate through all of the macroblocks that contain 1 or more

     * coded fragments */

    for (sb_y = 0; sb_y < s->y_superblock_height; sb_y++) {

        for (sb_x = 0; sb_x < s->y_superblock_width; sb_x++) {



        for (j = 0; j < 4; j++) {

            int mb_x = 2*sb_x +   (j>>1);

            int mb_y = 2*sb_y + (((j>>1)+j)&1);

            current_macroblock = mb_y * s->macroblock_width + mb_x;



            if (mb_x >= s->macroblock_width || mb_y >= s->macroblock_height ||

                (s->macroblock_coding[current_macroblock] == MODE_COPY))

                continue;



            switch (s->macroblock_coding[current_macroblock]) {



            case MODE_INTER_PLUS_MV:

            case MODE_GOLDEN_MV:

                /* all 6 fragments use the same motion vector */

                if (coding_mode == 0) {

                    motion_x[0] = motion_vector_table[get_vlc2(gb, s->motion_vector_vlc.table, 6, 2)];

                    motion_y[0] = motion_vector_table[get_vlc2(gb, s->motion_vector_vlc.table, 6, 2)];

                } else {

                    motion_x[0] = fixed_motion_vector_table[get_bits(gb, 6)];

                    motion_y[0] = fixed_motion_vector_table[get_bits(gb, 6)];

                }



                /* vector maintenance, only on MODE_INTER_PLUS_MV */

                if (s->macroblock_coding[current_macroblock] ==

                    MODE_INTER_PLUS_MV) {

                    prior_last_motion_x = last_motion_x;

                    prior_last_motion_y = last_motion_y;

                    last_motion_x = motion_x[0];

                    last_motion_y = motion_y[0];

                }

                break;



            case MODE_INTER_FOURMV:

                /* vector maintenance */

                prior_last_motion_x = last_motion_x;

                prior_last_motion_y = last_motion_y;



                /* fetch 4 vectors from the bitstream, one for each

                 * Y fragment, then average for the C fragment vectors */

                motion_x[4] = motion_y[4] = 0;

                for (k = 0; k < 4; k++) {

                    current_fragment = BLOCK_Y*s->fragment_width + BLOCK_X;

                    for (l = 0; l < s->coded_fragment_list_index; l++)

                        if (s->coded_fragment_list[l] == current_fragment)

                            break;

                    if (l < s->coded_fragment_list_index) {

                        if (coding_mode == 0) {

                            motion_x[k] = motion_vector_table[get_vlc2(gb, s->motion_vector_vlc.table, 6, 2)];

                            motion_y[k] = motion_vector_table[get_vlc2(gb, s->motion_vector_vlc.table, 6, 2)];

                        } else {

                            motion_x[k] = fixed_motion_vector_table[get_bits(gb, 6)];

                            motion_y[k] = fixed_motion_vector_table[get_bits(gb, 6)];

                        }

                        last_motion_x = motion_x[k];

                        last_motion_y = motion_y[k];

                    } else {

                        motion_x[k] = 0;

                        motion_y[k] = 0;

                    }

                    motion_x[4] += motion_x[k];

                    motion_y[4] += motion_y[k];

                }



                motion_x[5]=

                motion_x[4]= RSHIFT(motion_x[4], 2);

                motion_y[5]=

                motion_y[4]= RSHIFT(motion_y[4], 2);

                break;



            case MODE_INTER_LAST_MV:

                /* all 6 fragments use the last motion vector */

                motion_x[0] = last_motion_x;

                motion_y[0] = last_motion_y;



                /* no vector maintenance (last vector remains the

                 * last vector) */

                break;



            case MODE_INTER_PRIOR_LAST:

                /* all 6 fragments use the motion vector prior to the

                 * last motion vector */

                motion_x[0] = prior_last_motion_x;

                motion_y[0] = prior_last_motion_y;



                /* vector maintenance */

                prior_last_motion_x = last_motion_x;

                prior_last_motion_y = last_motion_y;

                last_motion_x = motion_x[0];

                last_motion_y = motion_y[0];

                break;



            default:

                /* covers intra, inter without MV, golden without MV */

                motion_x[0] = 0;

                motion_y[0] = 0;



                /* no vector maintenance */

                break;

            }



            /* assign the motion vectors to the correct fragments */

            for (k = 0; k < 4; k++) {

                current_fragment =

                    BLOCK_Y*s->fragment_width + BLOCK_X;

                if (s->macroblock_coding[current_macroblock] == MODE_INTER_FOURMV) {

                    s->all_fragments[current_fragment].motion_x = motion_x[k];

                    s->all_fragments[current_fragment].motion_y = motion_y[k];

                } else {

                    s->all_fragments[current_fragment].motion_x = motion_x[0];

                    s->all_fragments[current_fragment].motion_y = motion_y[0];

                }

            }

            for (k = 0; k < 2; k++) {

                current_fragment = s->fragment_start[k+1] +

                    mb_y*(s->fragment_width>>1) + mb_x;

                if (s->macroblock_coding[current_macroblock] == MODE_INTER_FOURMV) {

                    s->all_fragments[current_fragment].motion_x = motion_x[k+4];

                    s->all_fragments[current_fragment].motion_y = motion_y[k+4];

                } else {

                    s->all_fragments[current_fragment].motion_x = motion_x[0];

                    s->all_fragments[current_fragment].motion_y = motion_y[0];

                }

            }

        }

        }

    }



    return 0;

}
