static int kvm_sclp_service_call(S390CPU *cpu, struct kvm_run *run,

                                 uint16_t ipbh0)

{

    CPUS390XState *env = &cpu->env;

    uint64_t sccb;

    uint32_t code;

    int r = 0;



    cpu_synchronize_state(CPU(cpu));

    if (env->psw.mask & PSW_MASK_PSTATE) {

        enter_pgmcheck(cpu, PGM_PRIVILEGED);

        return 0;

    }

    sccb = env->regs[ipbh0 & 0xf];

    code = env->regs[(ipbh0 & 0xf0) >> 4];



    r = sclp_service_call(sccb, code);

    if (r < 0) {

        enter_pgmcheck(cpu, -r);

    }

    setcc(cpu, r);



    return 0;

}
