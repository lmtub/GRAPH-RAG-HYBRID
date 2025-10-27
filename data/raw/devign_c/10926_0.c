static void openrisc_sim_net_init(MemoryRegion *address_space,

                                  hwaddr base,

                                  hwaddr descriptors,

                                  qemu_irq irq, NICInfo *nd)

{

    DeviceState *dev;

    SysBusDevice *s;



    dev = qdev_create(NULL, "open_eth");

    qdev_set_nic_properties(dev, nd);

    qdev_init_nofail(dev);



    s = SYS_BUS_DEVICE(dev);

    sysbus_connect_irq(s, 0, irq);

    memory_region_add_subregion(address_space, base,

                                sysbus_mmio_get_region(s, 0));

    memory_region_add_subregion(address_space, descriptors,

                                sysbus_mmio_get_region(s, 1));

}
