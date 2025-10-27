static int decode_band_types(AACContext *ac, enum BandType band_type[120],

                             int band_type_run_end[120], GetBitContext *gb,

                             IndividualChannelStream *ics)

{

    int g, idx = 0;

    const int bits = (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) ? 3 : 5;

    for (g = 0; g < ics->num_window_groups; g++) {

        int k = 0;

        while (k < ics->max_sfb) {

            uint8_t sect_end = k;

            int sect_len_incr;

            int sect_band_type = get_bits(gb, 4);

            if (sect_band_type == 12) {

                av_log(ac->avctx, AV_LOG_ERROR, "invalid band type\n");

                return -1;

            }

            while ((sect_len_incr = get_bits(gb, bits)) == (1 << bits) - 1)

                sect_end += sect_len_incr;

            sect_end += sect_len_incr;

            if (get_bits_left(gb) < 0) {

                av_log(ac->avctx, AV_LOG_ERROR, overread_err);

                return -1;

            }

            if (sect_end > ics->max_sfb) {

                av_log(ac->avctx, AV_LOG_ERROR,

                       "Number of bands (%d) exceeds limit (%d).\n",

                       sect_end, ics->max_sfb);

                return -1;

            }

            for (; k < sect_end; k++) {

                band_type        [idx]   = sect_band_type;

                band_type_run_end[idx++] = sect_end;

            }

        }

    }

    return 0;

}
