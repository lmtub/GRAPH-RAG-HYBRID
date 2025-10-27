static coroutine_fn void test_multi_co_schedule_entry(void *opaque)

{

    g_assert(to_schedule[id] == NULL);

    atomic_mb_set(&to_schedule[id], qemu_coroutine_self());



    while (!atomic_mb_read(&now_stopping)) {

        int n;



        n = g_test_rand_int_range(0, NUM_CONTEXTS);

        schedule_next(n);

        qemu_coroutine_yield();



        g_assert(to_schedule[id] == NULL);

        atomic_mb_set(&to_schedule[id], qemu_coroutine_self());

    }

}
