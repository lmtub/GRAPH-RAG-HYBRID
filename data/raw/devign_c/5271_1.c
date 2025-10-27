int kvm_update_guest_debug(CPUState *env, unsigned long reinject_trap)

{

    struct kvm_set_guest_debug_data data;



    data.dbg.control = 0;

    if (env->singlestep_enabled)

        data.dbg.control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_SINGLESTEP;



    kvm_arch_update_guest_debug(env, &data.dbg);

    data.dbg.control |= reinject_trap;

    data.env = env;



    on_vcpu(env, kvm_invoke_set_guest_debug, &data);

    return data.err;

}
