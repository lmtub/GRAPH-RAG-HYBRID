static void bonito_spciconf_writew(void *opaque, target_phys_addr_t addr,

                                   uint32_t val)

{

    PCIBonitoState *s = opaque;

    PCIDevice *d = PCI_DEVICE(s);

    PCIHostState *phb = PCI_HOST_BRIDGE(s->pcihost);

    uint32_t pciaddr;

    uint16_t status;



    DPRINTF("bonito_spciconf_writew "TARGET_FMT_plx" val %x\n", addr, val);

    assert((addr & 0x1) == 0);



    pciaddr = bonito_sbridge_pciaddr(s, addr);



    if (pciaddr == 0xffffffff) {

        return;

    }



    /* set the pci address in s->config_reg */

    phb->config_reg = (pciaddr) | (1u << 31);

    pci_data_write(phb->bus, phb->config_reg, val, 2);



    /* clear PCI_STATUS_REC_MASTER_ABORT and PCI_STATUS_REC_TARGET_ABORT */

    status = pci_get_word(d->config + PCI_STATUS);

    status &= ~(PCI_STATUS_REC_MASTER_ABORT | PCI_STATUS_REC_TARGET_ABORT);

    pci_set_word(d->config + PCI_STATUS, status);

}
