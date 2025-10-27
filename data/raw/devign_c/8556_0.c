struct pxa2xx_i2c_s *pxa2xx_i2c_init(target_phys_addr_t base,

                qemu_irq irq, int ioregister)

{

    int iomemtype;

    struct pxa2xx_i2c_s *s = (struct pxa2xx_i2c_s *)

            i2c_slave_init(i2c_init_bus(), 0, sizeof(struct pxa2xx_i2c_s));



    s->base = base;

    s->irq = irq;

    s->slave.event = pxa2xx_i2c_event;

    s->slave.recv = pxa2xx_i2c_rx;

    s->slave.send = pxa2xx_i2c_tx;

    s->bus = i2c_init_bus();



    if (ioregister) {

        iomemtype = cpu_register_io_memory(0, pxa2xx_i2c_readfn,

                        pxa2xx_i2c_writefn, s);

        cpu_register_physical_memory(s->base & 0xfffff000, 0xfff, iomemtype);

    }



    register_savevm("pxa2xx_i2c", base, 0,

                    pxa2xx_i2c_save, pxa2xx_i2c_load, s);



    return s;

}
