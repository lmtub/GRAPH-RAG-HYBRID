static void fw_cfg_mem_realize(DeviceState *dev, Error **errp)

{

    FWCfgMemState *s = FW_CFG_MEM(dev);

    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);



    memory_region_init_io(&s->ctl_iomem, OBJECT(s), &fw_cfg_ctl_mem_ops,

                          FW_CFG(s), "fwcfg.ctl", FW_CFG_SIZE);

    sysbus_init_mmio(sbd, &s->ctl_iomem);



    memory_region_init_io(&s->data_iomem, OBJECT(s), &fw_cfg_data_mem_ops,

                          FW_CFG(s), "fwcfg.data",

                          fw_cfg_data_mem_ops.valid.max_access_size);

    sysbus_init_mmio(sbd, &s->data_iomem);

}
