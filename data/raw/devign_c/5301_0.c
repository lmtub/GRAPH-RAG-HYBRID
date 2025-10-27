static void calc_scales(DCAEncContext *c)

{

    int band, ch;



    for (band = 0; band < 32; band++)

        for (ch = 0; ch < c->fullband_channels; ch++)

            c->scale_factor[band][ch] = calc_one_scale(c->peak_cb[band][ch],

                                                       c->abits[band][ch],

                                                       &c->quant[band][ch]);



    if (c->lfe_channel)

        c->lfe_scale_factor = calc_one_scale(c->lfe_peak_cb, 11, &c->lfe_quant);

}
