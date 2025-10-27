static void breakpoint_invalidate(CPUState *env, target_ulong pc)

{

    target_ulong phys_addr;



    phys_addr = cpu_get_phys_page_debug(env, pc);

    tb_invalidate_phys_page_range(phys_addr, phys_addr + 1, 0);

}
