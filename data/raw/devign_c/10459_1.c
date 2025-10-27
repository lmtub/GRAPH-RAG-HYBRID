static inline int GET_TOK(TM2Context *ctx,int type) {

    if(ctx->tok_ptrs[type] >= ctx->tok_lens[type]) {

        av_log(ctx->avctx, AV_LOG_ERROR, "Read token from stream %i out of bounds (%i>=%i)\n", type, ctx->tok_ptrs[type], ctx->tok_lens[type]);

        return 0;

    }

    if(type <= TM2_MOT)

        return ctx->deltas[type][ctx->tokens[type][ctx->tok_ptrs[type]++]];

    return ctx->tokens[type][ctx->tok_ptrs[type]++];

}
