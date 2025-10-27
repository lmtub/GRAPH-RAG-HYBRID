static int pit_initfn(ISADevice *dev)

{

    PITState *pit = DO_UPCAST(PITState, dev, dev);

    PITChannelState *s;



    s = &pit->channels[0];

    /* the timer 0 is connected to an IRQ */

    s->irq_timer = qemu_new_timer(vm_clock, pit_irq_timer, s);

    s->irq = isa_reserve_irq(pit->irq);



    register_ioport_write(pit->iobase, 4, 1, pit_ioport_write, pit);

    register_ioport_read(pit->iobase, 3, 1, pit_ioport_read, pit);

    isa_init_ioport(dev, pit->iobase);



    return 0;

}
