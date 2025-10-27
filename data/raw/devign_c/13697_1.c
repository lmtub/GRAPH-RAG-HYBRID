static void test_tco_second_timeout_none(void)

{

    TestData td;

    const uint16_t ticks = TCO_SECS_TO_TICKS(256);

    QDict *ad;



    td.args = "-watchdog-action none";

    td.noreboot = false;

    test_init(&td);



    stop_tco(&td);

    clear_tco_status(&td);

    reset_on_second_timeout(true);

    set_tco_timeout(&td, ticks);

    load_tco(&td);

    start_tco(&td);

    clock_step(ticks * TCO_TICK_NSEC * 2);

    ad = get_watchdog_action();

    g_assert(!strcmp(qdict_get_str(ad, "action"), "none"));

    QDECREF(ad);



    stop_tco(&td);

    qtest_end();

}
