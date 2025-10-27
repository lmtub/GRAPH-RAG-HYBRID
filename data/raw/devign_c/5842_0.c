static void calc_masking(DCAEncContext *c, const int32_t *input)

{

    int i, k, band, ch, ssf;

    int32_t data[512];



    for (i = 0; i < 256; i++)

        for (ssf = 0; ssf < SUBSUBFRAMES; ssf++)

            c->masking_curve_cb[ssf][i] = -2047;



    for (ssf = 0; ssf < SUBSUBFRAMES; ssf++)

        for (ch = 0; ch < c->fullband_channels; ch++) {

            const int chi = c->channel_order_tab[ch];



            for (i = 0, k = 128 + 256 * ssf; k < 512; i++, k++)

                data[i] = c->history[k][ch];

            for (k -= 512; i < 512; i++, k++)

                data[i] = input[k * c->channels + chi];

            adjust_jnd(c->samplerate_index, data, c->masking_curve_cb[ssf]);

        }

    for (i = 0; i < 256; i++) {

        int32_t m = 2048;



        for (ssf = 0; ssf < SUBSUBFRAMES; ssf++)

            if (c->masking_curve_cb[ssf][i] < m)

                m = c->masking_curve_cb[ssf][i];

        c->eff_masking_curve_cb[i] = m;

    }



    for (band = 0; band < 32; band++) {

        c->band_masking_cb[band] = 2048;

        walk_band_low(c, band, 0, update_band_masking, NULL);

        walk_band_high(c, band, 0, update_band_masking, NULL);

    }

}
