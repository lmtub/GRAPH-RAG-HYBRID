static void ram_init(target_phys_addr_t addr, ram_addr_t RAM_size,

                     uint64_t max_mem)

{

    DeviceState *dev;

    SysBusDevice *s;

    RamDevice *d;



    /* allocate RAM */

    if ((uint64_t)RAM_size > max_mem) {

        fprintf(stderr,

                "qemu: Too much memory for this machine: %d, maximum %d\n",

                (unsigned int)(RAM_size / (1024 * 1024)),

                (unsigned int)(max_mem / (1024 * 1024)));

        exit(1);

    }

    dev = qdev_create(NULL, "memory");

    s = sysbus_from_qdev(dev);



    d = FROM_SYSBUS(RamDevice, s);

    d->size = RAM_size;

    qdev_init(dev);



    sysbus_mmio_map(s, 0, addr);

}
