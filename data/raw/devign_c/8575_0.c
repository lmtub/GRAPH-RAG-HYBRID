void kvm_arch_reset_vcpu(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;



    env->exception_injected = -1;

    env->interrupt_injected = -1;

    env->xcr0 = 1;

    if (kvm_irqchip_in_kernel()) {

        env->mp_state = cpu_is_bsp(cpu) ? KVM_MP_STATE_RUNNABLE :

                                          KVM_MP_STATE_UNINITIALIZED;

    } else {

        env->mp_state = KVM_MP_STATE_RUNNABLE;

    }



    if (cpu->hyperv_synic) {

        int i;

        for (i = 0; i < ARRAY_SIZE(env->msr_hv_synic_sint); i++) {

            env->msr_hv_synic_sint[i] = HV_SINT_MASKED;

        }

    }

}
