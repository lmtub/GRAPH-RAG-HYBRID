static void test_bh_flush(void)

{

    BHTestData data = { .n = 0 };

    data.bh = aio_bh_new(ctx, bh_test_cb, &data);



    qemu_bh_schedule(data.bh);

    g_assert_cmpint(data.n, ==, 0);



    wait_for_aio();

    g_assert_cmpint(data.n, ==, 1);



    g_assert(!aio_poll(ctx, false));

    g_assert_cmpint(data.n, ==, 1);

    qemu_bh_delete(data.bh);

}
