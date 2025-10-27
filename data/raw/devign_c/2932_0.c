static int init_quantization_noise(DCAEncContext *c, int noise)

{

    int ch, band, ret = 0;



    c->consumed_bits = 132 + 493 * c->fullband_channels;

    if (c->lfe_channel)

        c->consumed_bits += 72;



    /* attempt to guess the bit distribution based on the prevoius frame */

    for (ch = 0; ch < c->fullband_channels; ch++) {

        for (band = 0; band < 32; band++) {

            int snr_cb = c->peak_cb[band][ch] - c->band_masking_cb[band] - noise;



            if (snr_cb >= 1312) {

                c->abits[band][ch] = 26;

                ret |= USED_26ABITS;

            } else if (snr_cb >= 222) {

                c->abits[band][ch] = 8 + mul32(snr_cb - 222, 69000000);

                ret |= USED_NABITS;

            } else if (snr_cb >= 0) {

                c->abits[band][ch] = 2 + mul32(snr_cb, 106000000);

                ret |= USED_NABITS;

            } else {

                c->abits[band][ch] = 1;

                ret |= USED_1ABITS;

            }

        }

    }



    for (band = 0; band < 32; band++)

        for (ch = 0; ch < c->fullband_channels; ch++) {

            c->consumed_bits += bit_consumption[c->abits[band][ch]];

        }



    return ret;

}
