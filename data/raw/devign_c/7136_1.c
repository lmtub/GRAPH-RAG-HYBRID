static void test_aio_external_client(void)

{

    int i, j;



    for (i = 1; i < 3; i++) {

        EventNotifierTestData data = { .n = 0, .active = 10, .auto_set = true };

        event_notifier_init(&data.e, false);

        aio_set_event_notifier(ctx, &data.e, true, event_ready_cb);

        event_notifier_set(&data.e);

        for (j = 0; j < i; j++) {

            aio_disable_external(ctx);

        }

        for (j = 0; j < i; j++) {

            assert(!aio_poll(ctx, false));

            assert(event_notifier_test_and_clear(&data.e));

            event_notifier_set(&data.e);

            aio_enable_external(ctx);

        }

        assert(aio_poll(ctx, false));


        event_notifier_cleanup(&data.e);

    }

}