void qemu_init_vcpu(void *_env)

{

    CPUState *env = _env;

    int r;



    env->nr_cores = smp_cores;

    env->nr_threads = smp_threads;



    if (kvm_enabled()) {

        r = kvm_init_vcpu(env);

        if (r < 0) {

            fprintf(stderr, "kvm_init_vcpu failed: %s\n", strerror(-r));

            exit(1);

        }

        qemu_kvm_init_cpu_signals(env);

    } else {

        qemu_tcg_init_cpu_signals();

    }

}
