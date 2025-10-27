void mips_cpu_do_unaligned_access(CPUState *cs, vaddr addr,

                                  int is_write, int is_user, uintptr_t retaddr)

{

    MIPSCPU *cpu = MIPS_CPU(cs);

    CPUMIPSState *env = &cpu->env;



    env->CP0_BadVAddr = addr;

    do_raise_exception(env, (is_write == 1) ? EXCP_AdES : EXCP_AdEL, retaddr);

}
