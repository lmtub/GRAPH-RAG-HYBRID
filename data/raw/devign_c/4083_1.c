static void test_tco_max_timeout(void)

{

    TestData d;

    const uint16_t ticks = 0xffff;

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

    clock_step(((ticks & TCO_TMR_MASK) - 1) * TCO_TICK_NSEC);



    val = qpci_io_readw(d.dev, d.tco_io_base + TCO_RLD);

    g_assert_cmpint(val & TCO_RLD_MASK, ==, 1);

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS);

    ret = val & TCO_TIMEOUT ? 1 : 0;

    g_assert(ret == 0);

    clock_step(TCO_TICK_NSEC);

    val = qpci_io_readw(d.dev, d.tco_io_base + TCO1_STS);

    ret = val & TCO_TIMEOUT ? 1 : 0;

    g_assert(ret == 1);



    stop_tco(&d);

    qtest_end();

}
