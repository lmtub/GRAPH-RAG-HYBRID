void kvm_cpu_synchronize_state(CPUState *env)

{

    if (!env->kvm_vcpu_dirty)

        run_on_cpu(env, do_kvm_cpu_synchronize_state, env);

}
