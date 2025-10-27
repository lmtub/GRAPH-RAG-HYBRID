static int aic_decode_coeffs(GetBitContext *gb, int16_t *dst,

                             int band, int slice_width)

{

    int has_skips, coeff_type, coeff_bits, skip_type, skip_bits;

    const int num_coeffs = aic_num_band_coeffs[band];

    const uint8_t *scan = aic_scan[band];

    int mb, idx, val;



    has_skips  = get_bits1(gb);

    coeff_type = get_bits1(gb);

    coeff_bits = get_bits(gb, 3);



    if (has_skips) {

        skip_type = get_bits1(gb);

        skip_bits = get_bits(gb, 3);



        for (mb = 0; mb < slice_width; mb++) {

            idx = -1;

            do {

                GET_CODE(val, skip_type, skip_bits);

                idx += val + 1;

                if (idx >= num_coeffs)

                    break;

                GET_CODE(val, coeff_type, coeff_bits);

                val++;

                if (val >= 0x10000)

                    return AVERROR_INVALIDDATA;

                dst[scan[idx]] = val;

            } while (idx < num_coeffs - 1);

            dst += num_coeffs;

        }

    } else {

        for (mb = 0; mb < slice_width; mb++) {

            for (idx = 0; idx < num_coeffs; idx++) {

                GET_CODE(val, coeff_type, coeff_bits);

                if (val >= 0x10000)

                    return AVERROR_INVALIDDATA;

                dst[scan[idx]] = val;

            }

            dst += num_coeffs;

        }

    }

    return 0;

}
