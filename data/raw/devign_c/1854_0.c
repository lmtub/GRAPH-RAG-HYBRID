static PCIDevice *find_dev(sPAPREnvironment *spapr, uint64_t buid,

                           uint32_t config_addr)

{

    sPAPRPHBState *sphb = find_phb(spapr, buid);

    PCIHostState *phb = PCI_HOST_BRIDGE(sphb);

    int bus_num = (config_addr >> 16) & 0xFF;

    int devfn = (config_addr >> 8) & 0xFF;



    if (!phb) {

        return NULL;

    }



    return pci_find_device(phb->bus, bus_num, devfn);

}
