static int encode_slices(VC2EncContext *s)

{

    uint8_t *buf;

    int i, slice_x, slice_y, skip = 0;

    int bytes_left = 0;

    SliceArgs *enc_args = s->slice_args;



    int bytes_top[SLICE_REDIST_TOTAL] = {0};

    SliceArgs *top_loc[SLICE_REDIST_TOTAL] = {NULL};



    avpriv_align_put_bits(&s->pb);

    flush_put_bits(&s->pb);

    buf = put_bits_ptr(&s->pb);



    for (slice_y = 0; slice_y < s->num_y; slice_y++) {

        for (slice_x = 0; slice_x < s->num_x; slice_x++) {

            SliceArgs *args = &enc_args[s->num_x*slice_y + slice_x];

            bytes_left += args->bytes_left;

            for (i = 0; i < FFMIN(SLICE_REDIST_TOTAL, s->num_x*s->num_y); i++) {

                if (args->bytes > bytes_top[i]) {

                    bytes_top[i] = args->bytes;

                    top_loc[i] = args;

                    break;

                }

            }

        }

    }



    while (1) {

        int distributed = 0;

        for (i = 0; i < FFMIN(SLICE_REDIST_TOTAL, s->num_x*s->num_y); i++) {

            SliceArgs *args;

            int bits, bytes, diff, prev_bytes, new_idx;

            if (bytes_left <= 0)

                break;

            if (!top_loc[i] || !top_loc[i]->quant_idx)

                break;

            args = top_loc[i];

            prev_bytes = args->bytes;

            new_idx = av_clip(args->quant_idx - 1, 0, s->q_ceil);

            bits = count_hq_slice(s, args->cache, args->x, args->y, new_idx);

            bytes = FFALIGN((bits >> 3), s->size_scaler) + 4 + s->prefix_bytes;

            diff = bytes - prev_bytes;

            if ((bytes_left - diff) >= 0) {

                args->quant_idx = new_idx;

                args->bytes = bytes;

                bytes_left -= diff;

                distributed++;

            }

        }

        if (!distributed)

            break;

    }



    for (slice_y = 0; slice_y < s->num_y; slice_y++) {

        for (slice_x = 0; slice_x < s->num_x; slice_x++) {

            SliceArgs *args = &enc_args[s->num_x*slice_y + slice_x];

            init_put_bits(&args->pb, buf + skip, args->bytes);

            s->q_avg = (s->q_avg + args->quant_idx)/2;

            skip += args->bytes;

        }

    }



    s->avctx->execute(s->avctx, encode_hq_slice, enc_args, NULL, s->num_x*s->num_y,

                      sizeof(SliceArgs));



    skip_put_bytes(&s->pb, skip);



    return 0;

}
