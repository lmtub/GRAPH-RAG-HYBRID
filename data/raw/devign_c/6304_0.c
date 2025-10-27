static int on2avc_decode_band_types(On2AVCContext *c, GetBitContext *gb)

{

    int bits_per_sect = c->is_long ? 5 : 3;

    int esc_val = (1 << bits_per_sect) - 1;

    int num_bands = c->num_bands * c->num_windows;

    int band = 0, i, band_type, run_len, run;



    while (band < num_bands) {

        band_type = get_bits(gb, 4);

        run_len   = 1;

        do {

            run = get_bits(gb, bits_per_sect);

            run_len += run;

        } while (run == esc_val);

        if (band + run_len > num_bands) {

            av_log(c->avctx, AV_LOG_ERROR, "Invalid band type run\n");

            return AVERROR_INVALIDDATA;

        }

        for (i = band; i < band + run_len; i++) {

            c->band_type[i]    = band_type;

            c->band_run_end[i] = band + run_len;

        }

        band += run_len;

    }



    return 0;

}
