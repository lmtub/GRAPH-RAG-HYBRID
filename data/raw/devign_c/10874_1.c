static void test_tco_timeout(void)

{

    TestData d;

    const uint16_t ticks = TCO_SECS_TO_TICKS(4);

    uint32_t val;

    int ret;



    d.args = NULL;

    d.noreboot = true;

    test_init(&d);



    stop_tco(&d);

    clear_tco_status(&d);

    reset_on_second_timeout(false);

    set_tco_timeout(&d, ticks);

    load_tco(&d);

    start_tco(&d);

    clock_step(ticks * TCO_TICK_NSEC);



    /* test first timeout */

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS);

    ret = val & TCO_TIMEOUT ? 1 : 0;

    g_assert(ret == 1);



    /* test clearing timeout bit */

    val |= TCO_TIMEOUT;

    qpci_io_writew(d.dev, d.tco_io_base + TCO1_STS, val);

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS);

    ret = val & TCO_TIMEOUT ? 1 : 0;

    g_assert(ret == 0);



    /* test second timeout */

    clock_step(ticks * TCO_TICK_NSEC);

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS);

    ret = val & TCO_TIMEOUT ? 1 : 0;

    g_assert(ret == 1);

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO2_STS);

    ret = val & TCO_SECOND_TO_STS ? 1 : 0;

    g_assert(ret == 1);



    stop_tco(&d);

    qtest_end();

}
