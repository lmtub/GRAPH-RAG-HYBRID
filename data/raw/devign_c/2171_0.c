static void apic_init_ipi(APICState *s)

{

    int i;



    s->tpr = 0;

    s->spurious_vec = 0xff;

    s->log_dest = 0;

    s->dest_mode = 0xf;

    memset(s->isr, 0, sizeof(s->isr));

    memset(s->tmr, 0, sizeof(s->tmr));

    memset(s->irr, 0, sizeof(s->irr));

    for(i = 0; i < APIC_LVT_NB; i++)

        s->lvt[i] = 1 << 16; /* mask LVT */

    s->esr = 0;

    memset(s->icr, 0, sizeof(s->icr));

    s->divide_conf = 0;

    s->count_shift = 0;

    s->initial_count = 0;

    s->initial_count_load_time = 0;

    s->next_time = 0;



    cpu_reset(s->cpu_env);



    s->cpu_env->halted = !(s->apicbase & MSR_IA32_APICBASE_BSP);

}
