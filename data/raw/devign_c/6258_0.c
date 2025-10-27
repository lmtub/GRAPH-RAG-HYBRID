static void active_parameter_sets(HEVCContext *s)

{

    GetBitContext *gb = &s->HEVClc->gb;

    int num_sps_ids_minus1;

    int i;



    get_bits(gb, 4); // active_video_parameter_set_id

    get_bits(gb, 1); // self_contained_cvs_flag

    get_bits(gb, 1); // num_sps_ids_minus1

    num_sps_ids_minus1 = get_ue_golomb_long(gb); // num_sps_ids_minus1



    s->active_seq_parameter_set_id = get_ue_golomb_long(gb);



    for (i = 1; i <= num_sps_ids_minus1; i++)

        get_ue_golomb_long(gb); // active_seq_parameter_set_id[i]

}
