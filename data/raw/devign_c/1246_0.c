static void *qemu_dummy_cpu_thread_fn(void *arg)

{

#ifdef _WIN32

    fprintf(stderr, "qtest is not supported under Windows\n");

    exit(1);

#else

    CPUState *cpu = arg;

    sigset_t waitset;

    int r;



    qemu_mutex_lock_iothread();

    qemu_thread_get_self(cpu->thread);

    cpu->thread_id = qemu_get_thread_id();

    cpu->exception_index = -1;

    cpu->can_do_io = 1;



    sigemptyset(&waitset);

    sigaddset(&waitset, SIG_IPI);



    /* signal CPU creation */

    cpu->created = true;

    qemu_cond_signal(&qemu_cpu_cond);



    current_cpu = cpu;

    while (1) {

        current_cpu = NULL;

        qemu_mutex_unlock_iothread();

        do {

            int sig;

            r = sigwait(&waitset, &sig);

        } while (r == -1 && (errno == EAGAIN || errno == EINTR));

        if (r == -1) {

            perror("sigwait");

            exit(1);

        }

        qemu_mutex_lock_iothread();

        current_cpu = cpu;

        qemu_wait_io_event_common(cpu);

    }



    return NULL;

#endif

}
