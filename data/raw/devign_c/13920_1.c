static int lag_decode_arith_plane(LagarithContext *l, uint8_t *dst,

                                  int width, int height, int stride,

                                  const uint8_t *src, int src_size)

{

    int i = 0;

    int read = 0;

    uint32_t length;

    uint32_t offset = 1;

    int esc_count = src[0];

    GetBitContext gb;

    lag_rac rac;



    rac.avctx = l->avctx;

    l->zeros = 0;



    if (esc_count < 4) {

        length = width * height;

        if (esc_count && AV_RL32(src + 1) < length) {

            length = AV_RL32(src + 1);

            offset += 4;

        }



        init_get_bits(&gb, src + offset, src_size * 8);



        if (lag_read_prob_header(&rac, &gb) < 0)

            return -1;



        ff_lag_rac_init(&rac, &gb, length - stride);



        for (i = 0; i < height; i++)

            read += lag_decode_line(l, &rac, dst + (i * stride), width,

                                    stride, esc_count);



        if (read > length)

            av_log(l->avctx, AV_LOG_WARNING,

                   "Output more bytes than length (%d of %d)\n", read,

                   length);

    } else if (esc_count < 8) {

        esc_count -= 4;

        if (esc_count > 0) {

            /* Zero run coding only, no range coding. */

            for (i = 0; i < height; i++)

                src += lag_decode_zero_run_line(l, dst + (i * stride), src,

                                                width, esc_count);

        } else {

            /* Plane is stored uncompressed */

            for (i = 0; i < height; i++) {

                memcpy(dst + (i * stride), src, width);

                src += width;

            }

        }

    } else if (esc_count == 0xff) {

        /* Plane is a solid run of given value */

        for (i = 0; i < height; i++)

            memset(dst + i * stride, src[1], width);

        /* Do not apply prediction.

           Note: memset to 0 above, setting first value to src[1]

           and applying prediction gives the same result. */

        return 0;

    } else {

        av_log(l->avctx, AV_LOG_ERROR,

               "Invalid zero run escape code! (%#x)\n", esc_count);

        return -1;

    }



    for (i = 0; i < height; i++) {

        lag_pred_line(l, dst, width, stride, i);

        dst += stride;

    }



    return 0;

}
