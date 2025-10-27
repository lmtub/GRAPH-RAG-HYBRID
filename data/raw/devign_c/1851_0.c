static uint32_t bonito_spciconf_readb(void *opaque, target_phys_addr_t addr)

{

    PCIBonitoState *s = opaque;

    uint32_t pciaddr;

    uint16_t status;



    DPRINTF("bonito_spciconf_readb "TARGET_FMT_plx"  \n", addr);

    pciaddr = bonito_sbridge_pciaddr(s, addr);



    if (pciaddr == 0xffffffff) {

        return 0xff;

    }



    /* set the pci address in s->config_reg */

    s->pcihost->config_reg = (pciaddr) | (1u << 31);



    /* clear PCI_STATUS_REC_MASTER_ABORT and PCI_STATUS_REC_TARGET_ABORT */

    status = pci_get_word(s->dev.config + PCI_STATUS);

    status &= ~(PCI_STATUS_REC_MASTER_ABORT | PCI_STATUS_REC_TARGET_ABORT);

    pci_set_word(s->dev.config + PCI_STATUS, status);



    return pci_data_read(s->pcihost->bus, s->pcihost->config_reg, 1);

}
