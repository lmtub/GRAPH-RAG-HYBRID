static void calc_transform_coeffs_cpl(AC3DecodeContext *s)

{

    int bin, band, ch, band_end;



    bin = s->start_freq[CPL_CH];

    for (band = 0; band < s->num_cpl_bands; band++) {

        band_end = bin + s->cpl_band_sizes[band];

        for (; bin < band_end; bin++) {

            for (ch = 1; ch <= s->fbw_channels; ch++) {

                if (s->channel_in_cpl[ch]) {

                    s->fixed_coeffs[ch][bin] = ((int64_t)s->fixed_coeffs[CPL_CH][bin] *

                                                (int64_t)s->cpl_coords[ch][band]) >> 23;

                    if (ch == 2 && s->phase_flags[band])

                        s->fixed_coeffs[ch][bin] = -s->fixed_coeffs[ch][bin];

                }

            }

        }

    }

}
