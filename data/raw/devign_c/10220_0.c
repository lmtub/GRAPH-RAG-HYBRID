static void qemu_signal_lock(unsigned int msecs)

{

    qemu_mutex_lock(&qemu_fair_mutex);



    while (qemu_mutex_trylock(&qemu_global_mutex)) {

        qemu_thread_signal(tcg_cpu_thread, SIGUSR1);

        if (!qemu_mutex_timedlock(&qemu_global_mutex, msecs))

            break;

    }

    qemu_mutex_unlock(&qemu_fair_mutex);

}
