static void init_block_mapping(Vp3DecodeContext *s) 

{

    int i, j;

    signed int hilbert_walk_y[16];

    signed int hilbert_walk_c[16];

    signed int hilbert_walk_mb[4];



    int current_fragment = 0;

    int current_width = 0;

    int current_height = 0;

    int right_edge = 0;

    int bottom_edge = 0;

    int superblock_row_inc = 0;

    int *hilbert = NULL;

    int mapping_index = 0;



    int current_macroblock;

    int c_fragment;



    signed char travel_width[16] = {

         1,  1,  0, -1, 

         0,  0,  1,  0,

         1,  0,  1,  0,

         0, -1,  0,  1

    };



    signed char travel_height[16] = {

         0,  0,  1,  0,

         1,  1,  0, -1,

         0,  1,  0, -1,

        -1,  0, -1,  0

    };



    signed char travel_width_mb[4] = {

         1,  0,  1,  0

    };



    signed char travel_height_mb[4] = {

         0,  1,  0, -1

    };



    debug_vp3("  vp3: initialize block mapping tables\n");



    /* figure out hilbert pattern per these frame dimensions */

    hilbert_walk_y[0]  = 1;

    hilbert_walk_y[1]  = 1;

    hilbert_walk_y[2]  = s->fragment_width;

    hilbert_walk_y[3]  = -1;

    hilbert_walk_y[4]  = s->fragment_width;

    hilbert_walk_y[5]  = s->fragment_width;

    hilbert_walk_y[6]  = 1;

    hilbert_walk_y[7]  = -s->fragment_width;

    hilbert_walk_y[8]  = 1;

    hilbert_walk_y[9]  = s->fragment_width;

    hilbert_walk_y[10]  = 1;

    hilbert_walk_y[11] = -s->fragment_width;

    hilbert_walk_y[12] = -s->fragment_width;

    hilbert_walk_y[13] = -1;

    hilbert_walk_y[14] = -s->fragment_width;

    hilbert_walk_y[15] = 1;



    hilbert_walk_c[0]  = 1;

    hilbert_walk_c[1]  = 1;

    hilbert_walk_c[2]  = s->fragment_width / 2;

    hilbert_walk_c[3]  = -1;

    hilbert_walk_c[4]  = s->fragment_width / 2;

    hilbert_walk_c[5]  = s->fragment_width / 2;

    hilbert_walk_c[6]  = 1;

    hilbert_walk_c[7]  = -s->fragment_width / 2;

    hilbert_walk_c[8]  = 1;

    hilbert_walk_c[9]  = s->fragment_width / 2;

    hilbert_walk_c[10]  = 1;

    hilbert_walk_c[11] = -s->fragment_width / 2;

    hilbert_walk_c[12] = -s->fragment_width / 2;

    hilbert_walk_c[13] = -1;

    hilbert_walk_c[14] = -s->fragment_width / 2;

    hilbert_walk_c[15] = 1;



    hilbert_walk_mb[0] = 1;

    hilbert_walk_mb[1] = s->macroblock_width;

    hilbert_walk_mb[2] = 1;

    hilbert_walk_mb[3] = -s->macroblock_width;



    /* iterate through each superblock (all planes) and map the fragments */

    for (i = 0; i < s->superblock_count; i++) {

        debug_init("    superblock %d (u starts @ %d, v starts @ %d)\n",

            i, s->u_superblock_start, s->v_superblock_start);



        /* time to re-assign the limits? */

        if (i == 0) {



            /* start of Y superblocks */

            right_edge = s->fragment_width;

            bottom_edge = s->fragment_height;

            current_width = 0;

            current_height = 0;

            superblock_row_inc = 3 * s->fragment_width;

            hilbert = hilbert_walk_y;



            /* the first operation for this variable is to advance by 1 */

            current_fragment = -1;



        } else if (i == s->u_superblock_start) {



            /* start of U superblocks */

            right_edge = s->fragment_width / 2;

            bottom_edge = s->fragment_height / 2;

            current_width = 0;

            current_height = 0;

            superblock_row_inc = 3 * (s->fragment_width / 2);

            hilbert = hilbert_walk_c;



            /* the first operation for this variable is to advance by 1 */

            current_fragment = s->u_fragment_start - 1;



        } else if (i == s->v_superblock_start) {



            /* start of V superblocks */

            right_edge = s->fragment_width / 2;

            bottom_edge = s->fragment_height / 2;

            current_width = 0;

            current_height = 0;

            superblock_row_inc = 3 * (s->fragment_width / 2);

            hilbert = hilbert_walk_c;



            /* the first operation for this variable is to advance by 1 */

            current_fragment = s->v_fragment_start - 1;



        }



        if (current_width >= right_edge) {

            /* reset width and move to next superblock row */

            current_width = 0;

            current_height += 4;



            /* fragment is now at the start of a new superblock row */

            current_fragment += superblock_row_inc;

        }



        /* iterate through all 16 fragments in a superblock */

        for (j = 0; j < 16; j++) {

            current_fragment += hilbert[j];

            current_height += travel_height[j];



            /* check if the fragment is in bounds */

            if ((current_width <= right_edge) &&

                (current_height < bottom_edge)) {

                s->superblock_fragments[mapping_index] = current_fragment;

                debug_init("    mapping fragment %d to superblock %d, position %d\n", 

                    s->superblock_fragments[mapping_index], i, j);

            } else {

                s->superblock_fragments[mapping_index] = -1;

                debug_init("    superblock %d, position %d has no fragment\n", 

                    i, j);

            }



            current_width += travel_width[j];

            mapping_index++;

        }

    }



    /* initialize the superblock <-> macroblock mapping; iterate through

     * all of the Y plane superblocks to build this mapping */

    right_edge = s->macroblock_width;

    bottom_edge = s->macroblock_height;

    current_width = 0;

    current_height = 0;

    superblock_row_inc = s->macroblock_width;

    hilbert = hilbert_walk_mb;

    mapping_index = 0;

    current_macroblock = -1;

    for (i = 0; i < s->u_superblock_start; i++) {



        if (current_width >= right_edge) {

            /* reset width and move to next superblock row */

            current_width = 0;

            current_height += 2;



            /* macroblock is now at the start of a new superblock row */

            current_macroblock += superblock_row_inc;

        }



        /* iterate through each potential macroblock in the superblock */

        for (j = 0; j < 4; j++) {

            current_macroblock += hilbert_walk_mb[j];

            current_height += travel_height_mb[j];



            /* check if the macroblock is in bounds */

            if ((current_width <= right_edge) &&

                (current_height < bottom_edge)) {

                s->superblock_macroblocks[mapping_index] = current_macroblock;

                debug_init("    mapping macroblock %d to superblock %d, position %d\n",

                    s->superblock_macroblocks[mapping_index], i, j);

            } else {

                s->superblock_macroblocks[mapping_index] = -1;

                debug_init("    superblock %d, position %d has no macroblock\n",

                    i, j);

            }



            current_width += travel_width_mb[j];

            mapping_index++;

        }

    }



    /* initialize the macroblock <-> fragment mapping */

    current_fragment = 0;

    current_macroblock = 0;

    mapping_index = 0;

    for (i = 0; i < s->fragment_height; i += 2) {



        for (j = 0; j < s->fragment_width; j += 2) {



            debug_init("    macroblock %d contains fragments: ", current_macroblock);

            s->all_fragments[current_fragment].macroblock = current_macroblock;

            s->macroblock_fragments[mapping_index++] = current_fragment;

            debug_init("%d ", current_fragment);



            if (j + 1 < s->fragment_width) {

                s->all_fragments[current_fragment + 1].macroblock = current_macroblock;

                s->macroblock_fragments[mapping_index++] = current_fragment + 1;

                debug_init("%d ", current_fragment + 1);

            } else

                s->macroblock_fragments[mapping_index++] = -1;



            if (i + 1 < s->fragment_height) {

                s->all_fragments[current_fragment + s->fragment_width].macroblock = 

                    current_macroblock;

                s->macroblock_fragments[mapping_index++] = 

                    current_fragment + s->fragment_width;

                debug_init("%d ", current_fragment + s->fragment_width);

            } else

                s->macroblock_fragments[mapping_index++] = -1;



            if ((j + 1 < s->fragment_width) && (i + 1 < s->fragment_height)) {

                s->all_fragments[current_fragment + s->fragment_width + 1].macroblock = 

                    current_macroblock;

                s->macroblock_fragments[mapping_index++] = 

                    current_fragment + s->fragment_width + 1;

                debug_init("%d ", current_fragment + s->fragment_width + 1);

            } else

                s->macroblock_fragments[mapping_index++] = -1;



            /* C planes */

            c_fragment = s->u_fragment_start + 

                (i * s->fragment_width / 4) + (j / 2);

        s->all_fragments[c_fragment].macroblock = s->macroblock_count;

            s->macroblock_fragments[mapping_index++] = c_fragment;

            debug_init("%d ", c_fragment);



            c_fragment = s->v_fragment_start + 

                (i * s->fragment_width / 4) + (j / 2);

        s->all_fragments[c_fragment].macroblock = s->macroblock_count;

            s->macroblock_fragments[mapping_index++] = c_fragment;

            debug_init("%d ", c_fragment);



            debug_init("\n");



            if (j + 2 <= s->fragment_width)

                current_fragment += 2;

            else 

                current_fragment++;

            current_macroblock++;

        }



        current_fragment += s->fragment_width;

    }

}
