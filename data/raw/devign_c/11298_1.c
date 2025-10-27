target_phys_addr_t cpu_get_phys_page_debug (CPUState *env, target_ulong addr)

{

    mmu_ctx_t ctx;



    if (unlikely(get_physical_address(env, &ctx, addr, 0, ACCESS_INT, 1) != 0))

        return -1;



    return ctx.raddr & TARGET_PAGE_MASK;

}
