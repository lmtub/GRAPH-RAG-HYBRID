static void *rcu_update_perf_test(void *arg)

{

    long long n_updates_local = 0;



    rcu_register_thread();



    *(struct rcu_reader_data **)arg = &rcu_reader;

    atomic_inc(&nthreadsrunning);

    while (goflag == GOFLAG_INIT) {

        g_usleep(1000);

    }

    while (goflag == GOFLAG_RUN) {

        synchronize_rcu();

        n_updates_local++;

    }

    atomic_add(&n_updates, n_updates_local);



    rcu_unregister_thread();

    return NULL;

}
