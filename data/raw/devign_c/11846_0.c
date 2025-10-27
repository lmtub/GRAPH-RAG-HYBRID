static int decode_profile_tier_level(HEVCLocalContext *lc, PTL *ptl,

                                     int max_num_sub_layers)

{

    int i, j;

    GetBitContext *gb = &lc->gb;



    ptl->general_profile_space = get_bits(gb, 2);

    ptl->general_tier_flag     = get_bits1(gb);

    ptl->general_profile_idc   = get_bits(gb, 5);

    for (i = 0; i < 32; i++)

        ptl->general_profile_compatibility_flag[i] = get_bits1(gb);

    skip_bits1(gb); // general_progressive_source_flag

    skip_bits1(gb); // general_interlaced_source_flag

    skip_bits1(gb); // general_non_packed_constraint_flag

    skip_bits1(gb); // general_frame_only_constraint_flag

    if (get_bits(gb, 16) != 0) // XXX_reserved_zero_44bits[0..15]

        return -1;

    if (get_bits(gb, 16) != 0) // XXX_reserved_zero_44bits[16..31]

        return -1;

    if (get_bits(gb, 12) != 0) // XXX_reserved_zero_44bits[32..43]

        return -1;



    ptl->general_level_idc = get_bits(gb, 8);

    for (i = 0; i < max_num_sub_layers - 1; i++) {

        ptl->sub_layer_profile_present_flag[i] = get_bits1(gb);

        ptl->sub_layer_level_present_flag[i]   = get_bits1(gb);

    }

    if (max_num_sub_layers - 1 > 0)

        for (i = max_num_sub_layers - 1; i < 8; i++)

            skip_bits(gb, 2);  // reserved_zero_2bits[i]

    for (i = 0; i < max_num_sub_layers - 1; i++) {

        if (ptl->sub_layer_profile_present_flag[i]) {

            ptl->sub_layer_profile_space[i] = get_bits(gb, 2);

            ptl->sub_layer_tier_flag[i]     = get_bits(gb, 1);

            ptl->sub_layer_profile_idc[i]   = get_bits(gb, 5);

            for (j = 0; j < 32; j++)

                ptl->sub_layer_profile_compatibility_flags[i][j] = get_bits1(gb);

            skip_bits1(gb); // sub_layer_progressive_source_flag

            skip_bits1(gb); // sub_layer_interlaced_source_flag

            skip_bits1(gb); // sub_layer_non_packed_constraint_flag

            skip_bits1(gb); // sub_layer_frame_only_constraint_flag



            if (get_bits(gb, 16) != 0) // sub_layer_reserved_zero_44bits[0..15]

                return -1;

            if (get_bits(gb, 16) != 0) // sub_layer_reserved_zero_44bits[16..31]

                return -1;

            if (get_bits(gb, 12) != 0) // sub_layer_reserved_zero_44bits[32..43]

                return -1;

        }

        if (ptl->sub_layer_level_present_flag[i])

            ptl->sub_layer_level_idc[i] = get_bits(gb, 8);

    }

    return 0;

}
