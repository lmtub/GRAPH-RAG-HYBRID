int kvm_arch_put_registers(CPUState *env, int level)

{

    int ret;



    assert(cpu_is_stopped(env) || qemu_cpu_is_self(env));



    ret = kvm_getput_regs(env, 1);




    ret = kvm_put_xsave(env);




    ret = kvm_put_xcrs(env);




    ret = kvm_put_sregs(env);









    ret = kvm_put_msrs(env, level);




    if (level >= KVM_PUT_RESET_STATE) {

        ret = kvm_put_mp_state(env);





    ret = kvm_put_vcpu_events(env, level);




    ret = kvm_put_debugregs(env);




    /* must be last */

    ret = kvm_guest_debug_workarounds(env);




    return 0;
