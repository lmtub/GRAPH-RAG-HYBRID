PITState *pit_init(int base, qemu_irq irq)

{

    PITState *pit = &pit_state;

    PITChannelState *s;



    s = &pit->channels[0];

    /* the timer 0 is connected to an IRQ */

    s->irq_timer = qemu_new_timer(vm_clock, pit_irq_timer, s);

    s->irq = irq;



    vmstate_register(base, &vmstate_pit, pit);

    qemu_register_reset(pit_reset, pit);

    register_ioport_write(base, 4, 1, pit_ioport_write, pit);

    register_ioport_read(base, 3, 1, pit_ioport_read, pit);



    pit_reset(pit);



    return pit;

}
