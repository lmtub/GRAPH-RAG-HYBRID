static void apic_update_irq(APICState *s)

{

    int irrv, ppr;

    if (!(s->spurious_vec & APIC_SV_ENABLE))

        return;

    irrv = get_highest_priority_int(s->irr);

    if (irrv < 0)

        return;

    ppr = apic_get_ppr(s);

    if (ppr && (irrv & 0xf0) <= (ppr & 0xf0))

        return;

    cpu_interrupt(s->cpu_env, CPU_INTERRUPT_HARD);

}
