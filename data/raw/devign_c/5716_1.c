static unsigned int read_sbr_data(AACContext *ac, SpectralBandReplication *sbr,

                                  GetBitContext *gb, int id_aac)

{

    unsigned int cnt = get_bits_count(gb);



    if (id_aac == TYPE_SCE || id_aac == TYPE_CCE) {

        read_sbr_single_channel_element(ac, sbr, gb);

    } else if (id_aac == TYPE_CPE) {

        read_sbr_channel_pair_element(ac, sbr, gb);

    } else {

        av_log(ac->avccontext, AV_LOG_ERROR,

            "Invalid bitstream - cannot apply SBR to element type %d\n", id_aac);

        sbr->start = 0;

        return get_bits_count(gb) - cnt;

    }

    if (get_bits1(gb)) { // bs_extended_data

        int num_bits_left = get_bits(gb, 4); // bs_extension_size

        if (num_bits_left == 15)

            num_bits_left += get_bits(gb, 8); // bs_esc_count



        num_bits_left <<= 3;

        while (num_bits_left > 7) {

            num_bits_left -= 2;

            read_sbr_extension(ac, sbr, gb, get_bits(gb, 2), &num_bits_left); // bs_extension_id

        }

    }



    return get_bits_count(gb) - cnt;

}
