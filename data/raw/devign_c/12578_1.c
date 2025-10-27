static int pci_init_multifunction(PCIBus *bus, PCIDevice *dev)

{

    uint8_t slot = PCI_SLOT(dev->devfn);

    uint8_t func;



    if (dev->cap_present & QEMU_PCI_CAP_MULTIFUNCTION) {

        dev->config[PCI_HEADER_TYPE] |= PCI_HEADER_TYPE_MULTI_FUNCTION;

    }



    /*

     * multifunction bit is interpreted in two ways as follows.

     *   - all functions must set the bit to 1.

     *     Example: Intel X53

     *   - function 0 must set the bit, but the rest function (> 0)

     *     is allowed to leave the bit to 0.

     *     Example: PIIX3(also in qemu), PIIX4(also in qemu), ICH10,

     *

     * So OS (at least Linux) checks the bit of only function 0,

     * and doesn't see the bit of function > 0.

     *

     * The below check allows both interpretation.

     */

    if (PCI_FUNC(dev->devfn)) {

        PCIDevice *f0 = bus->devices[PCI_DEVFN(slot, 0)];

        if (f0 && !(f0->cap_present & QEMU_PCI_CAP_MULTIFUNCTION)) {

            /* function 0 should set multifunction bit */

            error_report("PCI: single function device can't be populated "

                         "in function %x.%x", slot, PCI_FUNC(dev->devfn));

            return -1;

        }

        return 0;

    }



    if (dev->cap_present & QEMU_PCI_CAP_MULTIFUNCTION) {

        return 0;

    }

    /* function 0 indicates single function, so function > 0 must be NULL */

    for (func = 1; func < PCI_FUNC_MAX; ++func) {

        if (bus->devices[PCI_DEVFN(slot, func)]) {

            error_report("PCI: %x.0 indicates single function, "

                         "but %x.%x is already populated.",

                         slot, slot, func);

            return -1;

        }

    }

    return 0;

}
