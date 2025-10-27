FWCfgState *fw_cfg_init_mem(hwaddr ctl_addr, hwaddr data_addr)

{

    DeviceState *dev;

    SysBusDevice *sbd;



    dev = qdev_create(NULL, TYPE_FW_CFG_MEM);

    qdev_prop_set_uint32(dev, "data_width",

                         fw_cfg_data_mem_ops.valid.max_access_size);



    fw_cfg_init1(dev);



    sbd = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(sbd, 0, ctl_addr);

    sysbus_mmio_map(sbd, 1, data_addr);



    return FW_CFG(dev);

}
