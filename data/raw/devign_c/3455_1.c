xilinx_pcie_init(MemoryRegion *sys_mem, uint32_t bus_nr,

                 hwaddr cfg_base, uint64_t cfg_size,

                 hwaddr mmio_base, uint64_t mmio_size,

                 qemu_irq irq, bool link_up)

{

    DeviceState *dev;

    MemoryRegion *cfg, *mmio;



    dev = qdev_create(NULL, TYPE_XILINX_PCIE_HOST);



    qdev_prop_set_uint32(dev, "bus_nr", bus_nr);

    qdev_prop_set_uint64(dev, "cfg_base", cfg_base);

    qdev_prop_set_uint64(dev, "cfg_size", cfg_size);

    qdev_prop_set_uint64(dev, "mmio_base", mmio_base);

    qdev_prop_set_uint64(dev, "mmio_size", mmio_size);

    qdev_prop_set_bit(dev, "link_up", link_up);



    qdev_init_nofail(dev);



    cfg = sysbus_mmio_get_region(SYS_BUS_DEVICE(dev), 0);

    memory_region_add_subregion_overlap(sys_mem, cfg_base, cfg, 0);



    mmio = sysbus_mmio_get_region(SYS_BUS_DEVICE(dev), 1);

    memory_region_add_subregion_overlap(sys_mem, 0, mmio, 0);



    qdev_connect_gpio_out_named(dev, "interrupt_out", 0, irq);



    return XILINX_PCIE_HOST(dev);

}
