static uint32_t pci_unin_config_readl (void *opaque,

                                       target_phys_addr_t addr)

{

    UNINState *s = opaque;



    return s->config_reg;

}
