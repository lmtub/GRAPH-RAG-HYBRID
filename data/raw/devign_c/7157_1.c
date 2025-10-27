static void parallel_isa_realizefn(DeviceState *dev, Error **errp)

{

    static int index;

    ISADevice *isadev = ISA_DEVICE(dev);

    ISAParallelState *isa = ISA_PARALLEL(dev);

    ParallelState *s = &isa->state;

    int base;

    uint8_t dummy;



    if (!s->chr) {

        error_setg(errp, "Can't create parallel device, empty char device");

        return;

    }



    if (isa->index == -1) {

        isa->index = index;

    }

    if (isa->index >= MAX_PARALLEL_PORTS) {

        error_setg(errp, "Max. supported number of parallel ports is %d.",

                   MAX_PARALLEL_PORTS);

        return;

    }

    if (isa->iobase == -1) {

        isa->iobase = isa_parallel_io[isa->index];

    }

    index++;



    base = isa->iobase;

    isa_init_irq(isadev, &s->irq, isa->isairq);

    qemu_register_reset(parallel_reset, s);



    if (qemu_chr_fe_ioctl(s->chr, CHR_IOCTL_PP_READ_STATUS, &dummy) == 0) {

        s->hw_driver = 1;

        s->status = dummy;

    }



    isa_register_portio_list(isadev, base,

                             (s->hw_driver

                              ? &isa_parallel_portio_hw_list[0]

                              : &isa_parallel_portio_sw_list[0]),

                             s, "parallel");

}
