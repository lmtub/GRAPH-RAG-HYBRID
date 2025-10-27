static void breakpoint_invalidate(CPUArchState *env, target_ulong pc)

{

    tb_invalidate_phys_addr(cpu_get_phys_page_debug(env, pc));

}
