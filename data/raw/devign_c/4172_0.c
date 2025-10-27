static void test_bh_schedule10(void)

{

    BHTestData data = { .n = 0, .max = 10 };

    data.bh = aio_bh_new(ctx, bh_test_cb, &data);



    qemu_bh_schedule(data.bh);

    g_assert_cmpint(data.n, ==, 0);



    g_assert(aio_poll(ctx, false));

    g_assert_cmpint(data.n, ==, 1);



    g_assert(aio_poll(ctx, true));

    g_assert_cmpint(data.n, ==, 2);



    wait_for_aio();

    g_assert_cmpint(data.n, ==, 10);



    g_assert(!aio_poll(ctx, false));

    g_assert_cmpint(data.n, ==, 10);

    qemu_bh_delete(data.bh);

}
