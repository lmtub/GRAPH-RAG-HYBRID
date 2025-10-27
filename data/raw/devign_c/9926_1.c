static void synthfilt_build_sb_samples (QDM2Context *q, GetBitContext *gb, int length, int sb_min, int sb_max)

{

    int sb, j, k, n, ch, run, channels;

    int joined_stereo, zero_encoding, chs;

    int type34_first;

    float type34_div = 0;

    float type34_predictor;

    float samples[10], sign_bits[16];



    if (length == 0) {

        // If no data use noise

        for (sb=sb_min; sb < sb_max; sb++)

            build_sb_samples_from_noise (q, sb);



        return;

    }



    for (sb = sb_min; sb < sb_max; sb++) {

        FIX_NOISE_IDX(q->noise_idx);



        channels = q->nb_channels;



        if (q->nb_channels <= 1 || sb < 12)

            joined_stereo = 0;

        else if (sb >= 24)

            joined_stereo = 1;

        else

            joined_stereo = (get_bits_left(gb) >= 1) ? get_bits1 (gb) : 0;



        if (joined_stereo) {

            if (get_bits_left(gb) >= 16)

                for (j = 0; j < 16; j++)

                    sign_bits[j] = get_bits1 (gb);



            for (j = 0; j < 64; j++)

                if (q->coding_method[1][sb][j] > q->coding_method[0][sb][j])

                    q->coding_method[0][sb][j] = q->coding_method[1][sb][j];



            fix_coding_method_array(sb, q->nb_channels, q->coding_method);

            channels = 1;

        }



        for (ch = 0; ch < channels; ch++) {

            zero_encoding = (get_bits_left(gb) >= 1) ? get_bits1(gb) : 0;

            type34_predictor = 0.0;

            type34_first = 1;



            for (j = 0; j < 128; ) {

                switch (q->coding_method[ch][sb][j / 2]) {

                    case 8:

                        if (get_bits_left(gb) >= 10) {

                            if (zero_encoding) {

                                for (k = 0; k < 5; k++) {

                                    if ((j + 2 * k) >= 128)

                                        break;

                                    samples[2 * k] = get_bits1(gb) ? dequant_1bit[joined_stereo][2 * get_bits1(gb)] : 0;

                                }

                            } else {

                                n = get_bits(gb, 8);

                                for (k = 0; k < 5; k++)

                                    samples[2 * k] = dequant_1bit[joined_stereo][random_dequant_index[n][k]];

                            }

                            for (k = 0; k < 5; k++)

                                samples[2 * k + 1] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        } else {

                            for (k = 0; k < 10; k++)

                                samples[k] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        }

                        run = 10;

                        break;



                    case 10:

                        if (get_bits_left(gb) >= 1) {

                            float f = 0.81;



                            if (get_bits1(gb))

                                f = -f;

                            f -= noise_samples[((sb + 1) * (j +5 * ch + 1)) & 127] * 9.0 / 40.0;

                            samples[0] = f;

                        } else {

                            samples[0] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        }

                        run = 1;

                        break;



                    case 16:

                        if (get_bits_left(gb) >= 10) {

                            if (zero_encoding) {

                                for (k = 0; k < 5; k++) {

                                    if ((j + k) >= 128)

                                        break;

                                    samples[k] = (get_bits1(gb) == 0) ? 0 : dequant_1bit[joined_stereo][2 * get_bits1(gb)];

                                }

                            } else {

                                n = get_bits (gb, 8);

                                for (k = 0; k < 5; k++)

                                    samples[k] = dequant_1bit[joined_stereo][random_dequant_index[n][k]];

                            }

                        } else {

                            for (k = 0; k < 5; k++)

                                samples[k] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        }

                        run = 5;

                        break;



                    case 24:

                        if (get_bits_left(gb) >= 7) {

                            n = get_bits(gb, 7);

                            for (k = 0; k < 3; k++)

                                samples[k] = (random_dequant_type24[n][k] - 2.0) * 0.5;

                        } else {

                            for (k = 0; k < 3; k++)

                                samples[k] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        }

                        run = 3;

                        break;



                    case 30:

                        if (get_bits_left(gb) >= 4)

                            samples[0] = type30_dequant[qdm2_get_vlc(gb, &vlc_tab_type30, 0, 1)];

                        else

                            samples[0] = SB_DITHERING_NOISE(sb,q->noise_idx);



                        run = 1;

                        break;



                    case 34:

                        if (get_bits_left(gb) >= 7) {

                            if (type34_first) {

                                type34_div = (float)(1 << get_bits(gb, 2));

                                samples[0] = ((float)get_bits(gb, 5) - 16.0) / 15.0;

                                type34_predictor = samples[0];

                                type34_first = 0;

                            } else {

                                samples[0] = type34_delta[qdm2_get_vlc(gb, &vlc_tab_type34, 0, 1)] / type34_div + type34_predictor;

                                type34_predictor = samples[0];

                            }

                        } else {

                            samples[0] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        }

                        run = 1;

                        break;



                    default:

                        samples[0] = SB_DITHERING_NOISE(sb,q->noise_idx);

                        run = 1;

                        break;

                }



                if (joined_stereo) {

                    float tmp[10][MPA_MAX_CHANNELS];



                    for (k = 0; k < run; k++) {

                        tmp[k][0] = samples[k];

                        tmp[k][1] = (sign_bits[(j + k) / 8]) ? -samples[k] : samples[k];

                    }

                    for (chs = 0; chs < q->nb_channels; chs++)

                        for (k = 0; k < run; k++)

                            if ((j + k) < 128)

                                q->sb_samples[chs][j + k][sb] = q->tone_level[chs][sb][((j + k)/2)] * tmp[k][chs];

                } else {

                    for (k = 0; k < run; k++)

                        if ((j + k) < 128)

                            q->sb_samples[ch][j + k][sb] = q->tone_level[ch][sb][(j + k)/2] * samples[k];

                }



                j += run;

            } // j loop

        } // channel loop

    } // subband loop

}
