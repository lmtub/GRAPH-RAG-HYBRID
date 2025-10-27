static void *sparc32_dma_init(target_phys_addr_t daddr, qemu_irq parent_irq,

                              void *iommu, qemu_irq *dev_irq)

{

    DeviceState *dev;

    SysBusDevice *s;



    dev = qdev_create(NULL, "sparc32_dma");

    qdev_prop_set_ptr(dev, "iommu_opaque", iommu);

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    sysbus_connect_irq(s, 0, parent_irq);

    *dev_irq = qdev_get_gpio_in(dev, 0);

    sysbus_mmio_map(s, 0, daddr);



    return s;

}
