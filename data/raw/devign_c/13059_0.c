S390PCIBusDevice *s390_pci_find_next_avail_dev(S390PCIBusDevice *pbdev)

{

    int idx = 0;

    S390PCIBusDevice *dev = NULL;

    S390pciState *s = s390_get_phb();



    if (pbdev) {

        idx = (pbdev->fh & FH_MASK_INDEX) + 1;

    }



    for (; idx < PCI_SLOT_MAX; idx++) {

        dev = s->pbdev[idx];

        if (dev && dev->state != ZPCI_FS_RESERVED) {

            return dev;

        }

    }



    return NULL;

}
