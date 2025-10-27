void hpet_init(qemu_irq *irq) {

    int i, iomemtype;

    HPETState *s;



    dprintf ("hpet_init\n");



    s = qemu_mallocz(sizeof(HPETState));

    hpet_statep = s;

    s->irqs = irq;

    for (i=0; i<HPET_NUM_TIMERS; i++) {

        HPETTimer *timer = &s->timer[i];

        timer->qemu_timer = qemu_new_timer(vm_clock, hpet_timer, timer);

    }

    hpet_reset(s);

    vmstate_register(-1, &vmstate_hpet, s);

    qemu_register_reset(hpet_reset, s);

    /* HPET Area */

    iomemtype = cpu_register_io_memory(hpet_ram_read,

                                       hpet_ram_write, s);

    cpu_register_physical_memory(HPET_BASE, 0x400, iomemtype);

}
