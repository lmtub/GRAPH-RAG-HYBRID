static uint64_t icp_pic_read(void *opaque, target_phys_addr_t offset,

                             unsigned size)

{

    icp_pic_state *s = (icp_pic_state *)opaque;



    switch (offset >> 2) {

    case 0: /* IRQ_STATUS */

        return s->level & s->irq_enabled;

    case 1: /* IRQ_RAWSTAT */

        return s->level;

    case 2: /* IRQ_ENABLESET */

        return s->irq_enabled;

    case 4: /* INT_SOFTSET */

        return s->level & 1;

    case 8: /* FRQ_STATUS */

        return s->level & s->fiq_enabled;

    case 9: /* FRQ_RAWSTAT */

        return s->level;

    case 10: /* FRQ_ENABLESET */

        return s->fiq_enabled;

    case 3: /* IRQ_ENABLECLR */

    case 5: /* INT_SOFTCLR */

    case 11: /* FRQ_ENABLECLR */

    default:

        printf ("icp_pic_read: Bad register offset 0x%x\n", (int)offset);

        return 0;

    }

}
