static void test_tco2_status_bits(void)

{

    TestData d;

    uint16_t ticks = 8;

    uint16_t val;

    int ret;



    d.args = "-watchdog-action none";

    test_init(&d);



    stop_tco(&d);

    clear_tco_status(&d);

    reset_on_second_timeout(true);

    set_tco_timeout(&d, ticks);

    load_tco(&d);

    start_tco(&d);

    clock_step(ticks * TCO_TICK_NSEC * 2);



    val = qpci_io_readw(d.dev, d.tco_io_base + TCO2_STS);

    ret = val & (TCO_SECOND_TO_STS | TCO_BOOT_STS) ? 1 : 0;

    g_assert(ret == 1);

    qpci_io_writew(d.dev, d.tco_io_base + TCO2_STS, val);

    g_assert_cmpint(qpci_io_readw(d.dev, d.tco_io_base + TCO2_STS), ==, 0);

    qtest_end();

}
