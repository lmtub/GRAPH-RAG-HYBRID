static void test_enabled(void)

{

    int i;



    throttle_config_init(&cfg);

    g_assert(!throttle_enabled(&cfg));



    for (i = 0; i < BUCKETS_COUNT; i++) {

        throttle_config_init(&cfg);

        set_cfg_value(false, i, 150);

        g_assert(throttle_enabled(&cfg));

    }



    for (i = 0; i < BUCKETS_COUNT; i++) {

        throttle_config_init(&cfg);

        set_cfg_value(false, i, -150);

        g_assert(!throttle_enabled(&cfg));

    }

}
