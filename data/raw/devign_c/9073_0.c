static void run_test(void)

{

    unsigned int remaining;

    int i;



    while (atomic_read(&n_ready_threads) != n_rw_threads + n_rz_threads) {

        cpu_relax();

    }

    atomic_mb_set(&test_start, true);

    do {

        remaining = sleep(duration);

    } while (remaining);

    atomic_mb_set(&test_stop, true);



    for (i = 0; i < n_rw_threads; i++) {

        qemu_thread_join(&rw_threads[i]);

    }

    for (i = 0; i < n_rz_threads; i++) {

        qemu_thread_join(&rz_threads[i]);

    }

}
