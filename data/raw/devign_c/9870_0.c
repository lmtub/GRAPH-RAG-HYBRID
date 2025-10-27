static S390PCIBusDevice *s390_pci_find_dev_by_target(const char *target)

{

    int i;

    S390PCIBusDevice *pbdev;

    S390pciState *s = s390_get_phb();



    if (!target) {

        return NULL;

    }



    for (i = 0; i < PCI_SLOT_MAX; i++) {

        pbdev = s->pbdev[i];

        if (!pbdev) {

            continue;

        }



        if (!strcmp(pbdev->target, target)) {

            return pbdev;

        }

    }



    return NULL;

}
