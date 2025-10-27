static void unmanageable_intercept(S390CPU *cpu, const char *str, int pswoffset)

{

    CPUState *cs = CPU(cpu);



    error_report("Unmanageable %s! CPU%i new PSW: 0x%016lx:%016lx",

                 str, cs->cpu_index, ldq_phys(cs->as, cpu->env.psa + pswoffset),

                 ldq_phys(cs->as, cpu->env.psa + pswoffset + 8));

    s390_cpu_halt(cpu);

    guest_panicked();

}
