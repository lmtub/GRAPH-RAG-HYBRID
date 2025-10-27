static bool s390_cpu_has_work(CPUState *cs)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;



    return (cs->interrupt_request & CPU_INTERRUPT_HARD) &&

           (env->psw.mask & PSW_MASK_EXT);

}
