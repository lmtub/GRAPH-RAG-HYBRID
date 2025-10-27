PXA2xxPCMCIAState *pxa2xx_pcmcia_init(MemoryRegion *sysmem,

                                      hwaddr base)

{

    DeviceState *dev;

    PXA2xxPCMCIAState *s;



    dev = qdev_create(NULL, TYPE_PXA2XX_PCMCIA);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, base);

    s = PXA2XX_PCMCIA(dev);



    if (base == 0x30000000) {

        s->slot.slot_string = "PXA PC Card Socket 1";

    } else {

        s->slot.slot_string = "PXA PC Card Socket 0";

    }



    qdev_init_nofail(dev);



    return s;

}
