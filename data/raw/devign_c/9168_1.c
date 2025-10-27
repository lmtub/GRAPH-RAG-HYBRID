static QPCIDevice *get_device(void)

{

    QPCIDevice *dev;

    QPCIBus *pcibus;



    pcibus = qpci_init_pc();

    dev = NULL;

    qpci_device_foreach(pcibus, 0x1af4, 0x1110, save_fn, &dev);

    g_assert(dev != NULL);



    return dev;

}
