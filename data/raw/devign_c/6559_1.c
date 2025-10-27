FWCfgState *fw_cfg_init_mem_wide(hwaddr ctl_addr,

                                 hwaddr data_addr, uint32_t data_width,

                                 hwaddr dma_addr, AddressSpace *dma_as)

{

    DeviceState *dev;

    SysBusDevice *sbd;

    FWCfgState *s;

    bool dma_requested = dma_addr && dma_as;



    dev = qdev_create(NULL, TYPE_FW_CFG_MEM);

    qdev_prop_set_uint32(dev, "data_width", data_width);

    if (!dma_requested) {

        qdev_prop_set_bit(dev, "dma_enabled", false);

    }



    fw_cfg_init1(dev);



    sbd = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(sbd, 0, ctl_addr);

    sysbus_mmio_map(sbd, 1, data_addr);



    s = FW_CFG(dev);



    if (s->dma_enabled) {

        s->dma_as = dma_as;

        s->dma_addr = 0;

        sysbus_mmio_map(sbd, 2, dma_addr);

    }



    return s;

}
