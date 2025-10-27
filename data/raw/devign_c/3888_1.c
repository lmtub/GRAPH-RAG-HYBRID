static DeviceState *slavio_intctl_init(target_phys_addr_t addr,

                                       target_phys_addr_t addrg,

                                       qemu_irq **parent_irq)

{

    DeviceState *dev;

    SysBusDevice *s;

    unsigned int i, j;



    dev = qdev_create(NULL, "slavio_intctl");

    qdev_init(dev);



    s = sysbus_from_qdev(dev);



    for (i = 0; i < MAX_CPUS; i++) {

        for (j = 0; j < MAX_PILS; j++) {

            sysbus_connect_irq(s, i * MAX_PILS + j, parent_irq[i][j]);

        }

    }

    sysbus_mmio_map(s, 0, addrg);

    for (i = 0; i < MAX_CPUS; i++) {

        sysbus_mmio_map(s, i + 1, addr + i * TARGET_PAGE_SIZE);

    }



    return dev;

}
