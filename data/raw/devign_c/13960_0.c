static void test_source_flush_event_notifier(void)

{

    EventNotifierTestData data = { .n = 0, .active = 10, .auto_set = true };

    event_notifier_init(&data.e, false);

    aio_set_event_notifier(ctx, &data.e, event_ready_cb);

    g_assert(g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 0);

    g_assert_cmpint(data.active, ==, 10);



    event_notifier_set(&data.e);

    g_assert(g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 1);

    g_assert_cmpint(data.active, ==, 9);

    g_assert(g_main_context_iteration(NULL, false));



    while (g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 10);

    g_assert_cmpint(data.active, ==, 0);

    g_assert(!g_main_context_iteration(NULL, false));



    aio_set_event_notifier(ctx, &data.e, NULL);

    while (g_main_context_iteration(NULL, false));

    event_notifier_cleanup(&data.e);

}
