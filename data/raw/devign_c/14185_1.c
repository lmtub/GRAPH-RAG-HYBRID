static void test_notify(void)

{

    g_assert(!aio_poll(ctx, false));

    aio_notify(ctx);

    g_assert(!aio_poll(ctx, true));

    g_assert(!aio_poll(ctx, false));

}
