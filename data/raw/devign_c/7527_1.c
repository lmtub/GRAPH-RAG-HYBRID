static int handle_hypercall(CPUState *env, struct kvm_run *run)

{

    int r;



    cpu_synchronize_state(env);

    r = s390_virtio_hypercall(env);

    kvm_arch_put_registers(env);



    return r;

}
