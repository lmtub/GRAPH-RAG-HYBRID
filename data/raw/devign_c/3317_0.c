void arm_sysctl_init(uint32_t base, uint32_t sys_id)

{

    DeviceState *dev;



    dev = qdev_create(NULL, "realview_sysctl");

    qdev_prop_set_uint32(dev, "sys_id", sys_id);

    qdev_init_nofail(dev);

    sysbus_mmio_map(sysbus_from_qdev(dev), 0, base);

}
