grlib_irqmp_writel(void *opaque, target_phys_addr_t addr, uint32_t value)

{

    IRQMP      *irqmp = opaque;

    IRQMPState *state;



    assert(irqmp != NULL);

    state = irqmp->state;

    assert(state != NULL);



    addr &= 0xff;



    /* global registers */

    switch (addr) {

    case LEVEL_OFFSET:

        value &= 0xFFFF << 1; /* clean up the value */

        state->level = value;

        return;



    case PENDING_OFFSET:

        /* Read Only */

        return;



    case FORCE0_OFFSET:

        /* This register is an "alias" for the force register of CPU 0 */



        value &= 0xFFFE; /* clean up the value */

        state->force[0] = value;

        grlib_irqmp_check_irqs(irqmp->state);

        return;



    case CLEAR_OFFSET:

        value &= ~1; /* clean up the value */

        state->pending &= ~value;

        return;



    case MP_STATUS_OFFSET:

        /* Read Only (no SMP support) */

        return;



    case BROADCAST_OFFSET:

        value &= 0xFFFE; /* clean up the value */

        state->broadcast = value;

        return;



    default:

        break;

    }



    /* mask registers */

    if (addr >= MASK_OFFSET && addr < FORCE_OFFSET) {

        int cpu = (addr - MASK_OFFSET) / 4;

        assert(cpu >= 0 && cpu < IRQMP_MAX_CPU);



        value &= ~1; /* clean up the value */

        state->mask[cpu] = value;

        grlib_irqmp_check_irqs(irqmp->state);

        return;

    }



    /* force registers */

    if (addr >= FORCE_OFFSET && addr < EXTENDED_OFFSET) {

        int cpu = (addr - FORCE_OFFSET) / 4;

        assert(cpu >= 0 && cpu < IRQMP_MAX_CPU);



        uint32_t force = value & 0xFFFE;

        uint32_t clear = (value >> 16) & 0xFFFE;

        uint32_t old   = state->force[cpu];



        state->force[cpu] = (old | force) & ~clear;

        grlib_irqmp_check_irqs(irqmp->state);

        return;

    }



    /* extended (not supported) */

    if (addr >= EXTENDED_OFFSET && addr < IRQMP_REG_SIZE) {

        int cpu = (addr - EXTENDED_OFFSET) / 4;

        assert(cpu >= 0 && cpu < IRQMP_MAX_CPU);



        value &= 0xF; /* clean up the value */

        state->extended[cpu] = value;

        return;

    }



    trace_grlib_irqmp_unknown_register("write", addr);

}
