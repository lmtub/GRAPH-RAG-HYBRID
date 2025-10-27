void apic_init_reset(DeviceState *dev)

{

    APICCommonState *s = APIC_COMMON(dev);

    APICCommonClass *info = APIC_COMMON_GET_CLASS(s);

    int i;



    if (!s) {

        return;

    }

    s->tpr = 0;

    s->spurious_vec = 0xff;

    s->log_dest = 0;

    s->dest_mode = 0xf;

    memset(s->isr, 0, sizeof(s->isr));

    memset(s->tmr, 0, sizeof(s->tmr));

    memset(s->irr, 0, sizeof(s->irr));

    for (i = 0; i < APIC_LVT_NB; i++) {

        s->lvt[i] = APIC_LVT_MASKED;

    }

    s->esr = 0;

    memset(s->icr, 0, sizeof(s->icr));

    s->divide_conf = 0;

    s->count_shift = 0;

    s->initial_count = 0;

    s->initial_count_load_time = 0;

    s->next_time = 0;

    s->wait_for_sipi = !cpu_is_bsp(s->cpu);



    if (s->timer) {

        timer_del(s->timer);

    }

    s->timer_expiry = -1;



    if (info->reset) {

        info->reset(s);

    }

}
