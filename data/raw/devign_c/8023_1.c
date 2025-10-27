static void calc_slice_sizes(VC2EncContext *s)

{

    int slice_x, slice_y;

    SliceArgs *enc_args = s->slice_args;



    for (slice_y = 0; slice_y < s->num_y; slice_y++) {

        for (slice_x = 0; slice_x < s->num_x; slice_x++) {

            SliceArgs *args = &enc_args[s->num_x*slice_y + slice_x];

            args->ctx = s;

            args->x = slice_x;

            args->y = slice_y;

            args->bits_ceil = s->slice_max_bytes << 3;

            args->bits_floor = s->slice_min_bytes << 3;

            memset(args->cache, 0, MAX_QUANT_INDEX*sizeof(*args->cache));

        }

    }



    /* Determine quantization indices and bytes per slice */

    s->avctx->execute(s->avctx, rate_control, enc_args, NULL, s->num_x*s->num_y,

                      sizeof(SliceArgs));

}
