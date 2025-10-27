static void perf_yield(void)

{

    unsigned int i, maxcycles;

    double duration;



    maxcycles = 100000000;

    i = maxcycles;

    Coroutine *coroutine = qemu_coroutine_create(yield_loop);



    g_test_timer_start();

    while (i > 0) {

        qemu_coroutine_enter(coroutine, &i);

    }

    duration = g_test_timer_elapsed();



    g_test_message("Yield %u iterations: %f s\n",

        maxcycles, duration);

}
