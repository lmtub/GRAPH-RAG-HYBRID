static int pte64_check (mmu_ctx_t *ctx,

                        target_ulong pte0, target_ulong pte1, int h, int rw)

{

    return _pte_check(ctx, 1, pte0, pte1, h, rw);

}
