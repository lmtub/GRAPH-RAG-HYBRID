static void bonito_spciconf_writel(void *opaque, target_phys_addr_t addr,

                                   uint32_t val)

{

    PCIBonitoState *s = opaque;

    uint32_t pciaddr;

    uint16_t status;



    DPRINTF("bonito_spciconf_writel "TARGET_FMT_plx" val %x \n", addr, val);

    assert((addr&0x3)==0);



    pciaddr = bonito_sbridge_pciaddr(s, addr);



    if (pciaddr == 0xffffffff) {

        return;

    }



    /* set the pci address in s->config_reg */

    s->pcihost->config_reg = (pciaddr) | (1u << 31);

    pci_data_write(s->pcihost->bus, s->pcihost->config_reg, val, 4);



    /* clear PCI_STATUS_REC_MASTER_ABORT and PCI_STATUS_REC_TARGET_ABORT */

    status = pci_get_word(s->dev.config + PCI_STATUS);

    status &= ~(PCI_STATUS_REC_MASTER_ABORT | PCI_STATUS_REC_TARGET_ABORT);

    pci_set_word(s->dev.config + PCI_STATUS, status);

}
