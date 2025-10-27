static void test_bh_delete_from_cb(void)

{

    BHTestData data1 = { .n = 0, .max = 1 };



    data1.bh = aio_bh_new(ctx, bh_delete_cb, &data1);



    qemu_bh_schedule(data1.bh);

    g_assert_cmpint(data1.n, ==, 0);



    wait_for_aio();

    g_assert_cmpint(data1.n, ==, data1.max);

    g_assert(data1.bh == NULL);



    g_assert(!aio_poll(ctx, false));

}
