static int find_pte64 (mmu_ctx_t *ctx, int h, int rw)

{

    return _find_pte(ctx, 1, h, rw);

}
