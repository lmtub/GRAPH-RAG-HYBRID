void alpha_cpu_unassigned_access(CPUState *cs, hwaddr addr,

                                 bool is_write, bool is_exec, int unused,

                                 unsigned size)

{

    AlphaCPU *cpu = ALPHA_CPU(cs);

    CPUAlphaState *env = &cpu->env;



    env->trap_arg0 = addr;

    env->trap_arg1 = is_write ? 1 : 0;

    dynamic_excp(env, 0, EXCP_MCHK, 0);

}
