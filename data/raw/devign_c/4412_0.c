static void dss_sp_unpack_coeffs(DssSpContext *p, const uint8_t *src)

{

    GetBitContext gb;

    DssSpFrame *fparam = &p->fparam;

    int i;

    int subframe_idx;

    uint32_t combined_pitch;

    uint32_t tmp;

    uint32_t pitch_lag;



    for (i = 0; i < DSS_SP_FRAME_SIZE; i += 2) {

        p->bits[i]     = src[i + 1];

        p->bits[i + 1] = src[i];

    }



    init_get_bits(&gb, p->bits, DSS_SP_FRAME_SIZE * 8);



    for (i = 0; i < 2; i++)

        fparam->filter_idx[i] = get_bits(&gb, 5);

    for (; i < 8; i++)

        fparam->filter_idx[i] = get_bits(&gb, 4);

    for (; i < 14; i++)

        fparam->filter_idx[i] = get_bits(&gb, 3);



    for (subframe_idx = 0; subframe_idx < 4; subframe_idx++) {

        fparam->sf_adaptive_gain[subframe_idx] = get_bits(&gb, 5);



        fparam->sf[subframe_idx].combined_pulse_pos = get_bits_long(&gb, 31);



        fparam->sf[subframe_idx].gain = get_bits(&gb, 6);



        for (i = 0; i < 7; i++)

            fparam->sf[subframe_idx].pulse_val[i] = get_bits(&gb, 3);

    }



    for (subframe_idx = 0; subframe_idx < 4; subframe_idx++) {

        unsigned int C72_binomials[PULSE_MAX] = {

            72, 2556, 59640, 1028790, 13991544, 156238908, 1473109704,

            3379081753

        };

        unsigned int combined_pulse_pos =

            fparam->sf[subframe_idx].combined_pulse_pos;

        int index = 6;



        if (combined_pulse_pos < C72_binomials[PULSE_MAX - 1]) {

            if (p->pulse_dec_mode) {

                int pulse, pulse_idx;

                pulse              = PULSE_MAX - 1;

                pulse_idx          = 71;

                combined_pulse_pos =

                    fparam->sf[subframe_idx].combined_pulse_pos;



                /* this part seems to be close to g723.1 gen_fcb_excitation()

                 * RATE_6300 */



                /* TODO: what is 7? size of subframe? */

                for (i = 0; i < 7; i++) {

                    for (;

                         combined_pulse_pos <

                         dss_sp_combinatorial_table[pulse][pulse_idx];

                         --pulse_idx)

                        ;

                    combined_pulse_pos -=

                        dss_sp_combinatorial_table[pulse][pulse_idx];

                    pulse--;

                    fparam->sf[subframe_idx].pulse_pos[i] = pulse_idx;

                }

            }

        } else {

            p->pulse_dec_mode = 0;



            /* why do we need this? */

            fparam->sf[subframe_idx].pulse_pos[6] = 0;



            for (i = 71; i >= 0; i--) {

                if (C72_binomials[index] <= combined_pulse_pos) {

                    combined_pulse_pos -= C72_binomials[index];



                    fparam->sf[subframe_idx].pulse_pos[6 - index] = i;



                    if (!index)

                        break;

                    --index;

                }

                --C72_binomials[0];

                if (index) {

                    int a;

                    for (a = 0; a < index; a++)

                        C72_binomials[a + 1] -= C72_binomials[a];

                }

            }

        }

    }



    combined_pitch = get_bits(&gb, 24);



    fparam->pitch_lag[0] = (combined_pitch % 151) + 36;



    combined_pitch /= 151;



    for (i = 1; i < SUBFRAMES; i++) {

        fparam->pitch_lag[i] = combined_pitch % 48;

        combined_pitch      /= 48;

    }



    pitch_lag = fparam->pitch_lag[0];

    for (i = 1; i < SUBFRAMES; i++) {

        if (pitch_lag > 162) {

            fparam->pitch_lag[i] += 162 - 23;

        } else {

            tmp = pitch_lag - 23;

            if (tmp < 36)

                tmp = 36;

            fparam->pitch_lag[i] += tmp;

        }

        pitch_lag = fparam->pitch_lag[i];

    }

}
