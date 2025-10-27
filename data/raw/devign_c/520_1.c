static bool use_goto_tb(DisasContext *ctx, target_ulong dest)

{

    /* Suppress goto_tb in the case of single-steping and IO.  */

    if ((ctx->base.tb->cflags & CF_LAST_IO) || ctx->base.singlestep_enabled) {

        return false;

    }

    return true;

}
