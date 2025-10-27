static int find_pte32 (mmu_ctx_t *ctx, int h, int rw)

{

    return _find_pte(ctx, 0, h, rw);

}
