static int read_sbr_grid(AACContext *ac, SpectralBandReplication *sbr,

                         GetBitContext *gb, SBRData *ch_data)

{

    int i;



    ch_data->bs_freq_res[0] = ch_data->bs_freq_res[ch_data->bs_num_env[1]];

    ch_data->bs_num_env[0] = ch_data->bs_num_env[1];

    ch_data->bs_amp_res = sbr->bs_amp_res_header;



    switch (ch_data->bs_frame_class = get_bits(gb, 2)) {

    case FIXFIX:

        ch_data->bs_num_env[1] = 1 << get_bits(gb, 2);

        if (ch_data->bs_num_env[1] == 1)

            ch_data->bs_amp_res = 0;



        ch_data->bs_pointer = 0;



        ch_data->bs_freq_res[1] = get_bits1(gb);

        for (i = 1; i < ch_data->bs_num_env[1]; i++)

            ch_data->bs_freq_res[i + 1] = ch_data->bs_freq_res[1];

        break;

    case FIXVAR:

        ch_data->bs_var_bord[1] = get_bits(gb, 2);

        ch_data->bs_num_rel[1]  = get_bits(gb, 2);

        ch_data->bs_num_env[1]  = ch_data->bs_num_rel[1] + 1;



        for (i = 0; i < ch_data->bs_num_rel[1]; i++)

            ch_data->bs_rel_bord[1][i] = 2 * get_bits(gb, 2) + 2;



        ch_data->bs_pointer = get_bits(gb, ceil_log2[ch_data->bs_num_env[1]]);



        for (i = 0; i < ch_data->bs_num_env[1]; i++)

            ch_data->bs_freq_res[ch_data->bs_num_env[1] - i] = get_bits1(gb);

        break;

    case VARFIX:

        ch_data->bs_var_bord[0] = get_bits(gb, 2);

        ch_data->bs_num_rel[0]  = get_bits(gb, 2);

        ch_data->bs_num_env[1]  = ch_data->bs_num_rel[0] + 1;



        for (i = 0; i < ch_data->bs_num_rel[0]; i++)

            ch_data->bs_rel_bord[0][i] = 2 * get_bits(gb, 2) + 2;



        ch_data->bs_pointer = get_bits(gb, ceil_log2[ch_data->bs_num_env[1]]);



        get_bits1_vector(gb, ch_data->bs_freq_res + 1, ch_data->bs_num_env[1]);

        break;

    case VARVAR:

        ch_data->bs_var_bord[0] = get_bits(gb, 2);

        ch_data->bs_var_bord[1] = get_bits(gb, 2);

        ch_data->bs_num_rel[0]  = get_bits(gb, 2);

        ch_data->bs_num_rel[1]  = get_bits(gb, 2);

        ch_data->bs_num_env[1]  = ch_data->bs_num_rel[0] + ch_data->bs_num_rel[1] + 1;



        for (i = 0; i < ch_data->bs_num_rel[0]; i++)

            ch_data->bs_rel_bord[0][i] = 2 * get_bits(gb, 2) + 2;

        for (i = 0; i < ch_data->bs_num_rel[1]; i++)

            ch_data->bs_rel_bord[1][i] = 2 * get_bits(gb, 2) + 2;



        ch_data->bs_pointer = get_bits(gb, ceil_log2[ch_data->bs_num_env[1]]);



        get_bits1_vector(gb, ch_data->bs_freq_res + 1, ch_data->bs_num_env[1]);

        break;

    }



    if (ch_data->bs_pointer > ch_data->bs_num_env[1] + 1) {

        av_log(ac->avccontext, AV_LOG_ERROR,

               "Invalid bitstream, bs_pointer points to a middle noise border outside the time borders table: %d\n",

               ch_data->bs_pointer);

        return -1;

    }

    if (ch_data->bs_frame_class == FIXFIX && ch_data->bs_num_env[1] > 4) {

        av_log(ac->avccontext, AV_LOG_ERROR,

               "Invalid bitstream, too many SBR envelopes in FIXFIX type SBR frame: %d\n",

               ch_data->bs_num_env[1]);

        return -1;

    }

    if (ch_data->bs_frame_class == VARVAR && ch_data->bs_num_env[1] > 5) {

        av_log(ac->avccontext, AV_LOG_ERROR,

               "Invalid bitstream, too many SBR envelopes in VARVAR type SBR frame: %d\n",

               ch_data->bs_num_env[1]);

        return -1;

    }



    ch_data->bs_num_noise = (ch_data->bs_num_env[1] > 1) + 1;



    return 0;

}
