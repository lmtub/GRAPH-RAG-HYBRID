static void test_source_timer_schedule(void)

{

    TimerTestData data = { .n = 0, .ctx = ctx, .ns = SCALE_MS * 750LL,

                           .max = 2,

                           .clock_type = QEMU_CLOCK_VIRTUAL };

    int pipefd[2];

    int64_t expiry;



    /* aio_poll will not block to wait for timers to complete unless it has

     * an fd to wait on. Fixing this breaks other tests. So create a dummy one.

     */

    g_assert(!qemu_pipe(pipefd));

    qemu_set_nonblock(pipefd[0]);

    qemu_set_nonblock(pipefd[1]);



    aio_set_fd_handler(ctx, pipefd[0],

                       dummy_io_handler_read, NULL, NULL);

    do {} while (g_main_context_iteration(NULL, false));



    aio_timer_init(ctx, &data.timer, data.clock_type,

                   SCALE_NS, timer_test_cb, &data);

    expiry = qemu_clock_get_ns(data.clock_type) +

        data.ns;

    timer_mod(&data.timer, expiry);



    g_assert_cmpint(data.n, ==, 0);



    g_usleep(1 * G_USEC_PER_SEC);

    g_assert_cmpint(data.n, ==, 0);



    g_assert(g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 1);



    /* The comment above was not kidding when it said this wakes up itself */

    do {

        g_assert(g_main_context_iteration(NULL, true));

    } while (qemu_clock_get_ns(data.clock_type) <= expiry);

    g_usleep(1 * G_USEC_PER_SEC);

    g_main_context_iteration(NULL, false);



    g_assert_cmpint(data.n, ==, 2);



    aio_set_fd_handler(ctx, pipefd[0], NULL, NULL, NULL);

    close(pipefd[0]);

    close(pipefd[1]);



    timer_del(&data.timer);

}
