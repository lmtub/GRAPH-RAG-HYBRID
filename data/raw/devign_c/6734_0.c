static void decorrelate_stereo_24(int32_t *buffer[MAX_CHANNELS],

                                  int32_t *buffer_out,

                                  int32_t *wasted_bits_buffer[MAX_CHANNELS],

                                  int wasted_bits,

                                  int numchannels, int numsamples,

                                  uint8_t interlacing_shift,

                                  uint8_t interlacing_leftweight)

{

    int i;



    if (numsamples <= 0)

        return;



    /* weighted interlacing */

    if (interlacing_leftweight) {

        for (i = 0; i < numsamples; i++) {

            int32_t a, b;



            a = buffer[0][i];

            b = buffer[1][i];



            a -= (b * interlacing_leftweight) >> interlacing_shift;

            b += a;



            if (wasted_bits) {

                b  = (b  << wasted_bits) | wasted_bits_buffer[0][i];

                a  = (a  << wasted_bits) | wasted_bits_buffer[1][i];

            }



            buffer_out[i * numchannels]     = b << 8;

            buffer_out[i * numchannels + 1] = a << 8;

        }

    } else {

        for (i = 0; i < numsamples; i++) {

            int32_t left, right;



            left  = buffer[0][i];

            right = buffer[1][i];



            if (wasted_bits) {

                left   = (left   << wasted_bits) | wasted_bits_buffer[0][i];

                right  = (right  << wasted_bits) | wasted_bits_buffer[1][i];

            }



            buffer_out[i * numchannels]     = left  << 8;

            buffer_out[i * numchannels + 1] = right << 8;

        }

    }

}
