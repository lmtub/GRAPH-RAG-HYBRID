static int handle_sigp(S390CPU *cpu, struct kvm_run *run, uint8_t ipa1)

{

    CPUS390XState *env = &cpu->env;

    const uint8_t r1 = ipa1 >> 4;

    const uint8_t r3 = ipa1 & 0x0f;

    int ret;

    uint8_t order;

    uint64_t *status_reg;

    uint64_t param;

    S390CPU *dst_cpu = NULL;



    cpu_synchronize_state(CPU(cpu));



    /* get order code */

    order = decode_basedisp_rs(env, run->s390_sieic.ipb, NULL)

        & SIGP_ORDER_MASK;

    status_reg = &env->regs[r1];

    param = (r1 % 2) ? env->regs[r1] : env->regs[r1 + 1];



    if (qemu_mutex_trylock(&qemu_sigp_mutex)) {

        ret = SIGP_CC_BUSY;

        goto out;

    }



    switch (order) {

    case SIGP_SET_ARCH:

        ret = sigp_set_architecture(cpu, param, status_reg);

        break;

    default:

        /* all other sigp orders target a single vcpu */

        dst_cpu = s390_cpu_addr2state(env->regs[r3]);

        ret = handle_sigp_single_dst(dst_cpu, order, param, status_reg);

    }





    trace_kvm_sigp_finished(order, CPU(cpu)->cpu_index,

                            dst_cpu ? CPU(dst_cpu)->cpu_index : -1, ret);



    if (ret >= 0) {

        setcc(cpu, ret);

        return 0;

    }



    return ret;

}