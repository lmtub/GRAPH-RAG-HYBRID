static void bonito_pciconf_writel(void *opaque, target_phys_addr_t addr,

                                  uint32_t val)

{

    PCIBonitoState *s = opaque;



    DPRINTF("bonito_pciconf_writel "TARGET_FMT_plx" val %x \n", addr, val);

    s->dev.config_write(&s->dev, addr, val, 4);

}
