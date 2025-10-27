static void gic_cpu_write(gic_state *s, int cpu, int offset, uint32_t value)

{

    switch (offset) {

    case 0x00: /* Control */

        s->cpu_enabled[cpu] = (value & 1);

        DPRINTF("CPU %d %sabled\n", cpu, s->cpu_enabled ? "En" : "Dis");

        break;

    case 0x04: /* Priority mask */

        s->priority_mask[cpu] = (value & 0xff);

        break;

    case 0x08: /* Binary Point */

        /* ??? Not implemented.  */

        break;

    case 0x10: /* End Of Interrupt */

        return gic_complete_irq(s, cpu, value & 0x3ff);

    default:

        hw_error("gic_cpu_write: Bad offset %x\n", (int)offset);

        return;

    }

    gic_update(s);

}
