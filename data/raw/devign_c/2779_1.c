static void free_ahci_device(QPCIDevice *dev)

{

    QPCIBus *pcibus = dev ? dev->bus : NULL;



    /* libqos doesn't have a function for this, so free it manually */

    g_free(dev);

    qpci_free_pc(pcibus);

}
