static void test_submit(void)

{

    WorkerTestData data = { .n = 0 };

    thread_pool_submit(pool, worker_cb, &data);

    qemu_aio_wait_all();

    g_assert_cmpint(data.n, ==, 1);

}
