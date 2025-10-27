static inline int ape_decode_value(APEContext *ctx, APERice *rice)

{

    int x, overflow;



    if (ctx->fileversion < 3990) {

        int tmpk;



        overflow = range_get_symbol(ctx, counts_3970, counts_diff_3970);



        if (overflow == (MODEL_ELEMENTS - 1)) {

            tmpk = range_decode_bits(ctx, 5);

            overflow = 0;

        } else

            tmpk = (rice->k < 1) ? 0 : rice->k - 1;



        if (tmpk <= 16)

            x = range_decode_bits(ctx, tmpk);

        else {

            x = range_decode_bits(ctx, 16);

            x |= (range_decode_bits(ctx, tmpk - 16) << 16);

        }

        x += overflow << tmpk;

    } else {

        int base, pivot;



        pivot = rice->ksum >> 5;

        if (pivot == 0)

            pivot = 1;



        overflow = range_get_symbol(ctx, counts_3980, counts_diff_3980);



        if (overflow == (MODEL_ELEMENTS - 1)) {

            overflow  = range_decode_bits(ctx, 16) << 16;

            overflow |= range_decode_bits(ctx, 16);

        }



        if (pivot < 0x10000) {

            base = range_decode_culfreq(ctx, pivot);

            range_decode_update(ctx, 1, base);

        } else {

            int base_hi = pivot, base_lo;

            int bbits = 0;



            while (base_hi & ~0xFFFF) {

                base_hi >>= 1;

                bbits++;

            }

            base_hi = range_decode_culfreq(ctx, base_hi + 1);

            range_decode_update(ctx, 1, base_hi);

            base_lo = range_decode_culfreq(ctx, 1 << bbits);

            range_decode_update(ctx, 1, base_lo);



            base = (base_hi << bbits) + base_lo;

        }



        x = base + overflow * pivot;

    }



    update_rice(rice, x);



    /* Convert to signed */

    if (x & 1)

        return (x >> 1) + 1;

    else

        return -(x >> 1);

}
