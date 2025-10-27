static void kvm_invoke_set_guest_debug(void *data)

{

    struct kvm_set_guest_debug_data *dbg_data = data;

    CPUState *env = dbg_data->env;



    if (env->kvm_vcpu_dirty) {

        kvm_arch_put_registers(env);

        env->kvm_vcpu_dirty = 0;

    }

    dbg_data->err = kvm_vcpu_ioctl(env, KVM_SET_GUEST_DEBUG, &dbg_data->dbg);

}
