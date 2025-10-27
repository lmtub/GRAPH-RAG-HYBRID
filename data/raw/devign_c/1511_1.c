void qemu_notify_event(void)

{

    CPUState *env = cpu_single_env;



    qemu_event_increment ();

    if (env) {

        cpu_exit(env);

    }

    if (next_cpu && env != next_cpu) {

        cpu_exit(next_cpu);

    }

    exit_request = 1;

}
