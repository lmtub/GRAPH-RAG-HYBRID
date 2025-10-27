static TranslationBlock *tb_alloc(target_ulong pc)

{

    TranslationBlock *tb;

    TBContext *ctx;



    assert_tb_locked();



    tb = tcg_tb_alloc(&tcg_ctx);

    if (unlikely(tb == NULL)) {

        return NULL;

    }

    ctx = &tcg_ctx.tb_ctx;

    if (unlikely(ctx->nb_tbs == ctx->tbs_size)) {

        ctx->tbs_size *= 2;

        ctx->tbs = g_renew(TranslationBlock *, ctx->tbs, ctx->tbs_size);

    }

    ctx->tbs[ctx->nb_tbs++] = tb;

    tb->pc = pc;

    tb->cflags = 0;

    tb->invalid = false;

    return tb;

}
