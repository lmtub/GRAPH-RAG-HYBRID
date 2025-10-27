static inline int ape_decode_value_3900(APEContext *ctx, APERice *rice)

{

    unsigned int x, overflow;

    int tmpk;



    overflow = range_get_symbol(ctx, counts_3970, counts_diff_3970);



    if (overflow == (MODEL_ELEMENTS - 1)) {

        tmpk = range_decode_bits(ctx, 5);

        overflow = 0;

    } else

        tmpk = (rice->k < 1) ? 0 : rice->k - 1;



    if (tmpk <= 16 || ctx->fileversion < 3910) {

        if (tmpk > 23) {

            av_log(ctx->avctx, AV_LOG_ERROR, "Too many bits: %d\n", tmpk);

            return AVERROR_INVALIDDATA;

        }

        x = range_decode_bits(ctx, tmpk);

    } else if (tmpk <= 32) {

        x = range_decode_bits(ctx, 16);

        x |= (range_decode_bits(ctx, tmpk - 16) << 16);

    } else {

        av_log(ctx->avctx, AV_LOG_ERROR, "Too many bits: %d\n", tmpk);

        return AVERROR_INVALIDDATA;

    }

    x += overflow << tmpk;



    update_rice(rice, x);



    /* Convert to signed */

    if (x & 1)

        return (x >> 1) + 1;

    else

        return -(x >> 1);

}
