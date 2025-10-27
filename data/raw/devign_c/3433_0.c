static int hpet_init(SysBusDevice *dev)

{

    HPETState *s = FROM_SYSBUS(HPETState, dev);

    int i, iomemtype;

    HPETTimer *timer;



    if (hpet_cfg.count == UINT8_MAX) {

        /* first instance */

        hpet_cfg.count = 0;

    }



    if (hpet_cfg.count == 8) {

        fprintf(stderr, "Only 8 instances of HPET is allowed\n");

        return -1;

    }



    s->hpet_id = hpet_cfg.count++;



    for (i = 0; i < HPET_NUM_IRQ_ROUTES; i++) {

        sysbus_init_irq(dev, &s->irqs[i]);

    }



    if (s->num_timers < HPET_MIN_TIMERS) {

        s->num_timers = HPET_MIN_TIMERS;

    } else if (s->num_timers > HPET_MAX_TIMERS) {

        s->num_timers = HPET_MAX_TIMERS;

    }

    for (i = 0; i < HPET_MAX_TIMERS; i++) {

        timer = &s->timer[i];

        timer->qemu_timer = qemu_new_timer(vm_clock, hpet_timer, timer);

        timer->tn = i;

        timer->state = s;

    }



    /* 64-bit main counter; LegacyReplacementRoute. */

    s->capability = 0x8086a001ULL;

    s->capability |= (s->num_timers - 1) << HPET_ID_NUM_TIM_SHIFT;

    s->capability |= ((HPET_CLK_PERIOD) << 32);



    isa_reserve_irq(RTC_ISA_IRQ);

    qdev_init_gpio_in(&dev->qdev, hpet_handle_rtc_irq, 1);



    /* HPET Area */

    iomemtype = cpu_register_io_memory(hpet_ram_read,

                                       hpet_ram_write, s,

                                       DEVICE_NATIVE_ENDIAN);

    sysbus_init_mmio(dev, 0x400, iomemtype);

    return 0;

}
