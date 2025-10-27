static int decode_nal_sei_frame_packing_arrangement(HEVCContext *s)

{

    GetBitContext *gb = &s->HEVClc->gb;



    get_ue_golomb(gb);                  // frame_packing_arrangement_id

    s->sei_frame_packing_present = !get_bits1(gb);



    if (s->sei_frame_packing_present) {

        s->frame_packing_arrangement_type = get_bits(gb, 7);

        s->quincunx_subsampling           = get_bits1(gb);

        s->content_interpretation_type    = get_bits(gb, 6);



        // the following skips spatial_flipping_flag frame0_flipped_flag

        // field_views_flag current_frame_is_frame0_flag

        // frame0_self_contained_flag frame1_self_contained_flag

        skip_bits(gb, 6);



        if (!s->quincunx_subsampling && s->frame_packing_arrangement_type != 5)

            skip_bits(gb, 16);  // frame[01]_grid_position_[xy]

        skip_bits(gb, 8);       // frame_packing_arrangement_reserved_byte

        skip_bits1(gb);         // frame_packing_arrangement_persistance_flag

    }

    skip_bits1(gb);             // upsampled_aspect_ratio_flag

    return 0;

}
