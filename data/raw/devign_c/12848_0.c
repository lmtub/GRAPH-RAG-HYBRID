static void slavio_set_timer_irq_cpu(void *opaque, int cpu, int level)

{

    SLAVIO_INTCTLState *s = opaque;



    DPRINTF("Set cpu %d local level %d\n", cpu, level);

    if (!s->cpu_envs[cpu])

        return;



    if (level) {

        s->intreg_pending[cpu] |= s->cputimer_bit;

    } else {

        s->intreg_pending[cpu] &= ~s->cputimer_bit;

    }



    slavio_check_interrupts(s);

}
