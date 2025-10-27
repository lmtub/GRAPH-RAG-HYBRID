static int tm2_read_stream(TM2Context *ctx, const uint8_t *buf, int stream_id) {

    int i;

    int cur = 0;

    int skip = 0;

    int len, toks;

    TM2Codes codes;



    /* get stream length in dwords */

    len = AV_RB32(buf); buf += 4; cur += 4;

    skip = len * 4 + 4;



    if(len == 0)

        return 4;



    toks = AV_RB32(buf); buf += 4; cur += 4;

    if(toks & 1) {

        len = AV_RB32(buf); buf += 4; cur += 4;

        if(len == TM2_ESCAPE) {

            len = AV_RB32(buf); buf += 4; cur += 4;

        }

        if(len > 0) {

            init_get_bits(&ctx->gb, buf, (skip - cur) * 8);

            if(tm2_read_deltas(ctx, stream_id) == -1)

                return -1;

            buf += ((get_bits_count(&ctx->gb) + 31) >> 5) << 2;

            cur += ((get_bits_count(&ctx->gb) + 31) >> 5) << 2;

        }

    }

    /* skip unused fields */

    if(AV_RB32(buf) == TM2_ESCAPE) {

        buf += 4; cur += 4; /* some unknown length - could be escaped too */

    }

    buf += 4; cur += 4;

    buf += 4; cur += 4; /* unused by decoder */



    init_get_bits(&ctx->gb, buf, (skip - cur) * 8);

    if(tm2_build_huff_table(ctx, &codes) == -1)

        return -1;

    buf += ((get_bits_count(&ctx->gb) + 31) >> 5) << 2;

    cur += ((get_bits_count(&ctx->gb) + 31) >> 5) << 2;



    toks >>= 1;

    /* check if we have sane number of tokens */

    if((toks < 0) || (toks > 0xFFFFFF)){

        av_log(ctx->avctx, AV_LOG_ERROR, "Incorrect number of tokens: %i\n", toks);

        tm2_free_codes(&codes);

        return -1;

    }

    ctx->tokens[stream_id] = av_realloc(ctx->tokens[stream_id], toks * sizeof(int));

    ctx->tok_lens[stream_id] = toks;

    len = AV_RB32(buf); buf += 4; cur += 4;

    if(len > 0) {

        init_get_bits(&ctx->gb, buf, (skip - cur) * 8);

        for(i = 0; i < toks; i++)

            ctx->tokens[stream_id][i] = tm2_get_token(&ctx->gb, &codes);

    } else {

        for(i = 0; i < toks; i++)

            ctx->tokens[stream_id][i] = codes.recode[0];

    }

    tm2_free_codes(&codes);



    return skip;

}
