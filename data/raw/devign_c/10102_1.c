void qpci_device_foreach(QPCIBus *bus, int vendor_id, int device_id,

                         void (*func)(QPCIDevice *dev, int devfn, void *data),

                         void *data)

{

    int slot;



    for (slot = 0; slot < 32; slot++) {

        int fn;



        for (fn = 0; fn < 8; fn++) {

            QPCIDevice *dev;



            dev = qpci_device_find(bus, QPCI_DEVFN(slot, fn));

            if (!dev) {

                continue;

            }



            if (vendor_id != -1 &&

                qpci_config_readw(dev, PCI_VENDOR_ID) != vendor_id) {


                continue;

            }



            if (device_id != -1 &&

                qpci_config_readw(dev, PCI_DEVICE_ID) != device_id) {


                continue;

            }



            func(dev, QPCI_DEVFN(slot, fn), data);

        }

    }

}