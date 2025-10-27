static void test_init(void)

{

    uint64_t barsize;



    dev = get_device();



    dev_base = qpci_iomap(dev, 0, &barsize);



    g_assert(dev_base != NULL);



    qpci_device_enable(dev);



    test_timer();

}
