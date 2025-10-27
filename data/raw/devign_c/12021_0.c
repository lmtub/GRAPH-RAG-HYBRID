static int handle_sigp(S390CPU *cpu, struct kvm_run *run, uint8_t ipa1)

{

    CPUS390XState *env = &cpu->env;

    uint8_t order_code;

    uint16_t cpu_addr;

    int r = -1;

    S390CPU *target_cpu;



    cpu_synchronize_state(CPU(cpu));



    /* get order code */

    order_code = decode_basedisp_rs(env, run->s390_sieic.ipb) & SIGP_ORDER_MASK;



    cpu_addr = env->regs[ipa1 & 0x0f];

    target_cpu = s390_cpu_addr2state(cpu_addr);

    if (target_cpu == NULL) {

        goto out;

    }



    switch (order_code) {

    case SIGP_START:

        r = kvm_s390_cpu_start(target_cpu);

        break;

    case SIGP_RESTART:

        r = kvm_s390_cpu_restart(target_cpu);

        break;

    case SIGP_SET_ARCH:

        /* make the caller panic */

        return -1;

    case SIGP_INITIAL_CPU_RESET:

        r = s390_cpu_initial_reset(target_cpu);

        break;

    default:

        fprintf(stderr, "KVM: unknown SIGP: 0x%x\n", order_code);

        break;

    }



out:

    setcc(cpu, r ? 3 : 0);

    return 0;

}
