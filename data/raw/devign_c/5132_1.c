static int unpack_vectors(Vp3DecodeContext *s, GetBitContext *gb)

{

    int i, j, k;

    int coding_mode;

    int motion_x[6];

    int motion_y[6];

    int last_motion_x = 0;

    int last_motion_y = 0;

    int prior_last_motion_x = 0;

    int prior_last_motion_y = 0;

    int current_macroblock;

    int current_fragment;



    debug_vp3("  vp3: unpacking motion vectors\n");

    if (s->keyframe) {



        debug_vp3("    keyframe-- there are no motion vectors\n");



    } else {



        memset(motion_x, 0, 6 * sizeof(int));

        memset(motion_y, 0, 6 * sizeof(int));



        /* coding mode 0 is the VLC scheme; 1 is the fixed code scheme */

        coding_mode = get_bits1(gb);

        debug_vectors("    using %s scheme for unpacking motion vectors\n",

            (coding_mode == 0) ? "VLC" : "fixed-length");



        /* iterate through all of the macroblocks that contain 1 or more

         * coded fragments */

        for (i = 0; i < s->u_superblock_start; i++) {



            for (j = 0; j < 4; j++) {

                current_macroblock = s->superblock_macroblocks[i * 4 + j];

                if ((current_macroblock == -1) ||

                    (s->macroblock_coding[current_macroblock] == MODE_COPY))

                    continue;

                if (current_macroblock >= s->macroblock_count) {

                    av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_vectors(): bad macroblock number (%d >= %d)\n",

                        current_macroblock, s->macroblock_count);

                    return 1;

                }



                current_fragment = s->macroblock_fragments[current_macroblock * 6];

                if (current_fragment >= s->fragment_count) {

                    av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_vectors(): bad fragment number (%d >= %d\n",

                        current_fragment, s->fragment_count);

                    return 1;

                }

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



                    for (k = 1; k < 6; k++) {

                        motion_x[k] = motion_x[0];

                        motion_y[k] = motion_y[0];

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

                        if (coding_mode == 0) {

                            motion_x[k] = motion_vector_table[get_vlc2(gb, s->motion_vector_vlc.table, 6, 2)];

                            motion_y[k] = motion_vector_table[get_vlc2(gb, s->motion_vector_vlc.table, 6, 2)];

                        } else {

                            motion_x[k] = fixed_motion_vector_table[get_bits(gb, 6)];

                            motion_y[k] = fixed_motion_vector_table[get_bits(gb, 6)];

                        }

                        last_motion_x = motion_x[k];

                        last_motion_y = motion_y[k];

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

                    for (k = 1; k < 6; k++) {

                        motion_x[k] = motion_x[0];

                        motion_y[k] = motion_y[0];

                    }



                    /* no vector maintenance (last vector remains the

                     * last vector) */

                    break;



                case MODE_INTER_PRIOR_LAST:

                    /* all 6 fragments use the motion vector prior to the

                     * last motion vector */

                    motion_x[0] = prior_last_motion_x;

                    motion_y[0] = prior_last_motion_y;

                    for (k = 1; k < 6; k++) {

                        motion_x[k] = motion_x[0];

                        motion_y[k] = motion_y[0];

                    }



                    /* vector maintenance */

                    prior_last_motion_x = last_motion_x;

                    prior_last_motion_y = last_motion_y;

                    last_motion_x = motion_x[0];

                    last_motion_y = motion_y[0];

                    break;



                default:

                    /* covers intra, inter without MV, golden without MV */

                    memset(motion_x, 0, 6 * sizeof(int));

                    memset(motion_y, 0, 6 * sizeof(int));



                    /* no vector maintenance */

                    break;

                }



                /* assign the motion vectors to the correct fragments */

                debug_vectors("    vectors for macroblock starting @ fragment %d (coding method %d):\n",

                    current_fragment,

                    s->macroblock_coding[current_macroblock]);

                for (k = 0; k < 6; k++) {

                    current_fragment =

                        s->macroblock_fragments[current_macroblock * 6 + k];

                    if (current_fragment == -1)

                        continue;

                    if (current_fragment >= s->fragment_count) {

                        av_log(s->avctx, AV_LOG_ERROR, "  vp3:unpack_vectors(): bad fragment number (%d >= %d)\n",

                            current_fragment, s->fragment_count);

                        return 1;

                    }

                    s->all_fragments[current_fragment].motion_x = motion_x[k];

                    s->all_fragments[current_fragment].motion_y = motion_y[k];

                    debug_vectors("    vector %d: fragment %d = (%d, %d)\n",

                        k, current_fragment, motion_x[k], motion_y[k]);

                }

            }

        }

    }



    return 0;

}
