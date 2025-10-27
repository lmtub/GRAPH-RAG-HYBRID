static void test_submit(void)

{

    WorkerTestData data = { .n = 0 };

    thread_pool_submit(worker_cb, &data);

    qemu_aio_flush();

    g_assert_cmpint(data.n, ==, 1);

}
