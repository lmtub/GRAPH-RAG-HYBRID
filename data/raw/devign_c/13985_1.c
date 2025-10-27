static void qemu_tcg_wait_io_event(void)

{

    CPUState *env;



    while (all_cpu_threads_idle()) {

       /* Start accounting real time to the virtual clock if the CPUs

          are idle.  */

        qemu_clock_warp(vm_clock);

        qemu_cond_wait(tcg_halt_cond, &qemu_global_mutex);

    }



    qemu_mutex_unlock(&qemu_global_mutex);



    /*

     * Users of qemu_global_mutex can be starved, having no chance

     * to acquire it since this path will get to it first.

     * So use another lock to provide fairness.

     */

    qemu_mutex_lock(&qemu_fair_mutex);

    qemu_mutex_unlock(&qemu_fair_mutex);



    qemu_mutex_lock(&qemu_global_mutex);



    for (env = first_cpu; env != NULL; env = env->next_cpu) {

        qemu_wait_io_event_common(env);

    }

}
