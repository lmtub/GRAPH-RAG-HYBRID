static uint64_t omap_sti_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    struct omap_sti_s *s = (struct omap_sti_s *) opaque;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0x00:	/* STI_REVISION */

        return 0x10;



    case 0x10:	/* STI_SYSCONFIG */

        return s->sysconfig;



    case 0x14:	/* STI_SYSSTATUS / STI_RX_STATUS / XTI_SYSSTATUS */

        return 0x00;



    case 0x18:	/* STI_IRQSTATUS */

        return s->irqst;



    case 0x1c:	/* STI_IRQSETEN / STI_IRQCLREN */

        return s->irqen;



    case 0x24:	/* STI_ER / STI_DR / XTI_TRACESELECT */

    case 0x28:	/* STI_RX_DR / XTI_RXDATA */

        /* TODO */

        return 0;



    case 0x2c:	/* STI_CLK_CTRL / XTI_SCLKCRTL */

        return s->clkcontrol;



    case 0x30:	/* STI_SERIAL_CFG / XTI_SCONFIG */

        return s->serial_config;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
