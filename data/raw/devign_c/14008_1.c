static void gen_sync(DisasContext *ctx)

{

    uint32_t l = (ctx->opcode >> 21) & 3;



    /*

     * We may need to check for a pending TLB flush.

     *

     * We do this on ptesync (l == 2) on ppc64 and any sync pn ppc32.

     *

     * Additionally, this can only happen in kernel mode however so

     * check MSR_PR as well.

     */

    if (((l == 2) || !(ctx->insns_flags & PPC_64B)) && !ctx->pr) {

        gen_check_tlb_flush(ctx);

    }

}
