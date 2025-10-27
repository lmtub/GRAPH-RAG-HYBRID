void ppc_slb_invalidate_one (CPUPPCState *env, uint64_t T0)

{

    /* XXX: TODO */

    tlb_flush(env, 1);

}
