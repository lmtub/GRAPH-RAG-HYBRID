static int on2avc_decode_band_scales(On2AVCContext *c, GetBitContext *gb)

{

    int w, w2, b, scale, first = 1;

    int band_off = 0;



    for (w = 0; w < c->num_windows; w++) {

        if (!c->grouping[w]) {

            memcpy(c->band_scales + band_off,

                   c->band_scales + band_off - c->num_bands,

                   c->num_bands * sizeof(*c->band_scales));

            band_off += c->num_bands;

            continue;

        }

        for (b = 0; b < c->num_bands; b++) {

            if (!c->band_type[band_off]) {

                int all_zero = 1;

                for (w2 = w + 1; w2 < c->num_windows; w2++) {

                    if (c->grouping[w2])

                        break;

                    if (c->band_type[w2 * c->num_bands + b]) {

                        all_zero = 0;

                        break;

                    }

                }

                if (all_zero) {

                    c->band_scales[band_off++] = 0;

                    continue;

                }

            }

            if (first) {

                scale = get_bits(gb, 7);

                first = 0;

            } else {

                scale += get_vlc2(gb, c->scale_diff.table, 9, 3) - 60;

            }

            if (scale < 0 || scale > 128) {

                av_log(c->avctx, AV_LOG_ERROR, "Invalid scale value %d\n",

                       scale);

                return AVERROR_INVALIDDATA;

            }

            c->band_scales[band_off++] = c->scale_tab[scale];

        }

    }



    return 0;

}
