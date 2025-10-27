S390PCIBusDevice *s390_pci_find_dev_by_fid(uint32_t fid)

{

    S390PCIBusDevice *pbdev;

    int i;

    S390pciState *s = s390_get_phb();



    for (i = 0; i < PCI_SLOT_MAX; i++) {

        pbdev = s->pbdev[i];

        if (pbdev && pbdev->fid == fid) {

            return pbdev;

        }

    }



    return NULL;

}
