static int ppc_hash64_get_physical_address(CPUPPCState *env,

                                           struct mmu_ctx_hash64 *ctx,

                                           target_ulong eaddr, int rw,

                                           int access_type)

{

    bool real_mode = (access_type == ACCESS_CODE && msr_ir == 0)

        || (access_type != ACCESS_CODE && msr_dr == 0);



    if (real_mode) {

        ctx->raddr = eaddr & 0x0FFFFFFFFFFFFFFFULL;

        ctx->prot = PAGE_READ | PAGE_EXEC | PAGE_WRITE;

        return 0;

    } else {

        return get_segment64(env, ctx, eaddr, rw, access_type);

    }

}
