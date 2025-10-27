static void test_tco_ticks_counter(void)

{

    TestData d;

    uint16_t ticks = TCO_SECS_TO_TICKS(8);

    uint16_t rld;



    d.args = NULL;

    d.noreboot = true;

    test_init(&d);



    stop_tco(&d);

    clear_tco_status(&d);

    reset_on_second_timeout(false);

    set_tco_timeout(&d, ticks);

    load_tco(&d);

    start_tco(&d);



    do {

        rld = qpci_io_readw(d.dev, d.tco_io_bar, TCO_RLD) & TCO_RLD_MASK;

        g_assert_cmpint(rld, ==, ticks);

        clock_step(TCO_TICK_NSEC);

        ticks--;

    } while (!(qpci_io_readw(d.dev, d.tco_io_bar, TCO1_STS) & TCO_TIMEOUT));



    stop_tco(&d);

    qtest_end();

}
