static int decode_display_orientation(H264Context *h)

{

    h->sei_display_orientation_present = !get_bits1(&h->gb);



    if (h->sei_display_orientation_present) {

        h->sei_hflip = get_bits1(&h->gb);     // hor_flip

        h->sei_vflip = get_bits1(&h->gb);     // ver_flip



        h->sei_anticlockwise_rotation = get_bits(&h->gb, 16);

        get_ue_golomb(&h->gb);  // display_orientation_repetition_period

        skip_bits1(&h->gb);     // display_orientation_extension_flag

    }



    return 0;

}
