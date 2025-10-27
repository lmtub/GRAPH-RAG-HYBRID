void kvm_arch_reset_vcpu(CPUX86State *env)

{

    env->exception_injected = -1;

    env->interrupt_injected = -1;

    env->xcr0 = 1;

    if (kvm_irqchip_in_kernel()) {

        env->mp_state = cpu_is_bsp(env) ? KVM_MP_STATE_RUNNABLE :

                                          KVM_MP_STATE_UNINITIALIZED;

    } else {

        env->mp_state = KVM_MP_STATE_RUNNABLE;

    }

}
