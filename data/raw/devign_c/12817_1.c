static void gen_isync(DisasContext *ctx)

{

    /*

     * We need to check for a pending TLB flush. This can only happen in

     * kernel mode however so check MSR_PR

     */

    if (!ctx->pr) {

        gen_check_tlb_flush(ctx);

    }

    gen_stop_exception(ctx);

}
