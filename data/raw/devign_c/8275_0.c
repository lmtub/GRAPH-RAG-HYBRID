static int decode_2(SANMVideoContext *ctx)

{

    int cx, cy, ret;



    for (cy = 0; cy != ctx->aligned_height; cy += 8) {

        for (cx = 0; cx != ctx->aligned_width; cx += 8) {

            if (ret = codec2subblock(ctx, cx, cy, 8))

                return ret;

        }

    }



    return 0;

}
