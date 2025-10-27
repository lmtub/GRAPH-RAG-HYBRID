static void test_submit_aio(void)

{

    WorkerTestData data = { .n = 0, .ret = -EINPROGRESS };

    data.aiocb = thread_pool_submit_aio(pool, worker_cb, &data,

                                        done_cb, &data);



    /* The callbacks are not called until after the first wait.  */

    active = 1;

    g_assert_cmpint(data.ret, ==, -EINPROGRESS);

    qemu_aio_wait_all();

    g_assert_cmpint(active, ==, 0);

    g_assert_cmpint(data.n, ==, 1);

    g_assert_cmpint(data.ret, ==, 0);

}
