DeviceState *sysbus_create_varargs(const char *name,

                                   target_phys_addr_t addr, ...)

{

    DeviceState *dev;

    SysBusDevice *s;

    va_list va;

    qemu_irq irq;

    int n;



    dev = qdev_create(NULL, name);

    s = sysbus_from_qdev(dev);

    qdev_init(dev);

    if (addr != (target_phys_addr_t)-1) {

        sysbus_mmio_map(s, 0, addr);

    }

    va_start(va, addr);

    n = 0;

    while (1) {

        irq = va_arg(va, qemu_irq);

        if (!irq) {

            break;

        }

        sysbus_connect_irq(s, n, irq);

        n++;

    }

    return dev;

}
