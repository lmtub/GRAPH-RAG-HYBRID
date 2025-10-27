static inline void start_exclusive(void)

{

    CPUState *other;

    pthread_mutex_lock(&exclusive_lock);

    exclusive_idle();



    pending_cpus = 1;

    /* Make all other cpus stop executing.  */

    for (other = first_cpu; other; other = other->next_cpu) {

        if (other->running) {

            pending_cpus++;

            cpu_interrupt(other, CPU_INTERRUPT_EXIT);

        }

    }

    if (pending_cpus > 1) {

        pthread_cond_wait(&exclusive_cond, &exclusive_lock);

    }

}
