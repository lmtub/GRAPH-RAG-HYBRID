int s390_virtio_hypercall(CPUS390XState *env)

{

    s390_virtio_fn fn = s390_diag500_table[env->regs[1]];



    if (!fn) {

        return -EINVAL;

    }



    return fn(&env->regs[2]);

}
