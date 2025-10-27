static uint32_t qpi_mem_readl(void *opaque, target_phys_addr_t addr)

{

    CPUState *env;



    env = cpu_single_env;

    if (!env)

        return 0;

    return env->eflags & (IF_MASK | IOPL_MASK);

}
