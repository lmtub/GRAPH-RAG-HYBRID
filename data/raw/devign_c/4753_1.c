static void read_sbr_channel_pair_element(AACContext *ac,

                                          SpectralBandReplication *sbr,

                                          GetBitContext *gb)

{

    if (get_bits1(gb))    // bs_data_extra

        skip_bits(gb, 8); // bs_reserved



    if ((sbr->bs_coupling = get_bits1(gb))) {

        read_sbr_grid(ac, sbr, gb, &sbr->data[0]);

        copy_sbr_grid(&sbr->data[1], &sbr->data[0]);

        read_sbr_dtdf(sbr, gb, &sbr->data[0]);

        read_sbr_dtdf(sbr, gb, &sbr->data[1]);

        read_sbr_invf(sbr, gb, &sbr->data[0]);

        memcpy(sbr->data[1].bs_invf_mode[1], sbr->data[1].bs_invf_mode[0], sizeof(sbr->data[1].bs_invf_mode[0]));

        memcpy(sbr->data[1].bs_invf_mode[0], sbr->data[0].bs_invf_mode[0], sizeof(sbr->data[1].bs_invf_mode[0]));

        read_sbr_envelope(sbr, gb, &sbr->data[0], 0);

        read_sbr_noise(sbr, gb, &sbr->data[0], 0);

        read_sbr_envelope(sbr, gb, &sbr->data[1], 1);

        read_sbr_noise(sbr, gb, &sbr->data[1], 1);

    } else {

        read_sbr_grid(ac, sbr, gb, &sbr->data[0]);

        read_sbr_grid(ac, sbr, gb, &sbr->data[1]);

        read_sbr_dtdf(sbr, gb, &sbr->data[0]);

        read_sbr_dtdf(sbr, gb, &sbr->data[1]);

        read_sbr_invf(sbr, gb, &sbr->data[0]);

        read_sbr_invf(sbr, gb, &sbr->data[1]);

        read_sbr_envelope(sbr, gb, &sbr->data[0], 0);

        read_sbr_envelope(sbr, gb, &sbr->data[1], 1);

        read_sbr_noise(sbr, gb, &sbr->data[0], 0);

        read_sbr_noise(sbr, gb, &sbr->data[1], 1);

    }



    if ((sbr->data[0].bs_add_harmonic_flag = get_bits1(gb)))

        get_bits1_vector(gb, sbr->data[0].bs_add_harmonic, sbr->n[1]);

    if ((sbr->data[1].bs_add_harmonic_flag = get_bits1(gb)))

        get_bits1_vector(gb, sbr->data[1].bs_add_harmonic, sbr->n[1]);

}
