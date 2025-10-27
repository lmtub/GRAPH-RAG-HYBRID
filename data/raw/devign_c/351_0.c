static void qdm2_fft_decode_tones (QDM2Context *q, int duration, GetBitContext *gb, int b)

{

    int channel, stereo, phase, exp;

    int local_int_4,  local_int_8,  stereo_phase,  local_int_10;

    int local_int_14, stereo_exp, local_int_20, local_int_28;

    int n, offset;



    local_int_4 = 0;

    local_int_28 = 0;

    local_int_20 = 2;

    local_int_8 = (4 - duration);

    local_int_10 = 1 << (q->group_order - duration - 1);

    offset = 1;



    while (1) {

        if (q->superblocktype_2_3) {

            while ((n = qdm2_get_vlc(gb, &vlc_tab_fft_tone_offset[local_int_8], 1, 2)) < 2) {

                offset = 1;

                if (n == 0) {

                    local_int_4 += local_int_10;

                    local_int_28 += (1 << local_int_8);

                } else {

                    local_int_4 += 8*local_int_10;

                    local_int_28 += (8 << local_int_8);

                }

            }

            offset += (n - 2);

        } else {

            offset += qdm2_get_vlc(gb, &vlc_tab_fft_tone_offset[local_int_8], 1, 2);

            while (offset >= (local_int_10 - 1)) {

                offset += (1 - (local_int_10 - 1));

                local_int_4  += local_int_10;

                local_int_28 += (1 << local_int_8);

            }

        }



        if (local_int_4 >= q->group_size)

            return;



        local_int_14 = (offset >> local_int_8);

        if (local_int_14 >= FF_ARRAY_ELEMS(fft_level_index_table))

            return;



        if (q->nb_channels > 1) {

            channel = get_bits1(gb);

            stereo = get_bits1(gb);

        } else {

            channel = 0;

            stereo = 0;

        }



        exp = qdm2_get_vlc(gb, (b ? &fft_level_exp_vlc : &fft_level_exp_alt_vlc), 0, 2);

        exp += q->fft_level_exp[fft_level_index_table[local_int_14]];

        exp = (exp < 0) ? 0 : exp;



        phase = get_bits(gb, 3);

        stereo_exp = 0;

        stereo_phase = 0;



        if (stereo) {

            stereo_exp = (exp - qdm2_get_vlc(gb, &fft_stereo_exp_vlc, 0, 1));

            stereo_phase = (phase - qdm2_get_vlc(gb, &fft_stereo_phase_vlc, 0, 1));

            if (stereo_phase < 0)

                stereo_phase += 8;

        }



        if (q->frequency_range > (local_int_14 + 1)) {

            int sub_packet = (local_int_20 + local_int_28);



            qdm2_fft_init_coefficient(q, sub_packet, offset, duration, channel, exp, phase);

            if (stereo)

                qdm2_fft_init_coefficient(q, sub_packet, offset, duration, (1 - channel), stereo_exp, stereo_phase);

        }



        offset++;

    }

}
