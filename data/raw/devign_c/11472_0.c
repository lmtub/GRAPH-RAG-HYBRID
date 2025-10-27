static void put_subframe_samples(DCAEncContext *c, int ss, int band, int ch)

{

    if (c->abits[band][ch] <= 7) {

        int sum, i, j;

        for (i = 0; i < 8; i += 4) {

            sum = 0;

            for (j = 3; j >= 0; j--) {

                sum *= ff_dca_quant_levels[c->abits[band][ch]];

                sum += c->quantized[ss * 8 + i + j][band][ch];

                sum += (ff_dca_quant_levels[c->abits[band][ch]] - 1) / 2;

            }

            put_bits(&c->pb, bit_consumption[c->abits[band][ch]] / 4, sum);

        }

    } else {

        int i;

        for (i = 0; i < 8; i++) {

            int bits = bit_consumption[c->abits[band][ch]] / 16;

            put_sbits(&c->pb, bits, c->quantized[ss * 8 + i][band][ch]);

        }

    }

}
