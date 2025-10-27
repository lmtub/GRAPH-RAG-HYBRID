static int tm2_read_deltas(TM2Context *ctx, int stream_id)

{

    int d, mb;

    int i, v;



    d  = get_bits(&ctx->gb, 9);

    mb = get_bits(&ctx->gb, 5);



    av_assert2(mb < 32);

    if ((d < 1) || (d > TM2_DELTAS) || (mb < 1)) {

        av_log(ctx->avctx, AV_LOG_ERROR, "Incorrect delta table: %i deltas x %i bits\n", d, mb);

        return AVERROR_INVALIDDATA;

    }



    for (i = 0; i < d; i++) {

        v = get_bits_long(&ctx->gb, mb);

        if (v & (1 << (mb - 1)))

            ctx->deltas[stream_id][i] = v - (1 << mb);

        else

            ctx->deltas[stream_id][i] = v;

    }

    for (; i < TM2_DELTAS; i++)

        ctx->deltas[stream_id][i] = 0;



    return 0;

}
