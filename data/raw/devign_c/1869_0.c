static int _do_rematrixing(AC3DecodeContext *ctx, int start, int end)

{

    float tmp0, tmp1;



    while (start < end) {

        tmp0 = ctx->samples[start];

        tmp1 = (ctx->samples + 256)[start];

        ctx->samples[start] = tmp0 + tmp1;

        (ctx->samples + 256)[start] = tmp0 - tmp1;

        start++;

    }



    return 0;

}
