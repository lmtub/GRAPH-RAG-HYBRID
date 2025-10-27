static void resume_all_vcpus(void)

{

    CPUState *penv = first_cpu;



    while (penv) {

        penv->stop = 0;

        penv->stopped = 0;

        qemu_thread_signal(penv->thread, SIGUSR1);

        qemu_cpu_kick(penv);

        penv = (CPUState *)penv->next_cpu;

    }

}
