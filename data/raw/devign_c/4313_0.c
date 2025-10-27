static uint64_t imx_avic_read(void *opaque,

                             target_phys_addr_t offset, unsigned size)

{

    IMXAVICState *s = (IMXAVICState *)opaque;





    DPRINTF("read(offset = 0x%x)\n", offset >> 2);

    switch (offset >> 2) {

    case 0: /* INTCNTL */

        return s->intcntl;



    case 1: /* Normal Interrupt Mask Register, NIMASK */

        return s->intmask;



    case 2: /* Interrupt Enable Number Register, INTENNUM */

    case 3: /* Interrupt Disable Number Register, INTDISNUM */

        return 0;



    case 4: /* Interrupt Enabled Number Register High */

        return s->enabled >> 32;



    case 5: /* Interrupt Enabled Number Register Low */

        return s->enabled & 0xffffffffULL;



    case 6: /* Interrupt Type Register High */

        return s->is_fiq >> 32;



    case 7: /* Interrupt Type Register Low */

        return s->is_fiq & 0xffffffffULL;



    case 8: /* Normal Interrupt Priority Register 7 */

    case 9: /* Normal Interrupt Priority Register 6 */

    case 10:/* Normal Interrupt Priority Register 5 */

    case 11:/* Normal Interrupt Priority Register 4 */

    case 12:/* Normal Interrupt Priority Register 3 */

    case 13:/* Normal Interrupt Priority Register 2 */

    case 14:/* Normal Interrupt Priority Register 1 */

    case 15:/* Normal Interrupt Priority Register 0 */

        return s->prio[15-(offset>>2)];



    case 16: /* Normal interrupt vector and status register */

    {

        /*

         * This returns the highest priority

         * outstanding interrupt.  Where there is more than

         * one pending IRQ with the same priority,

         * take the highest numbered one.

         */

        uint64_t flags = s->pending & s->enabled & ~s->is_fiq;

        int i;

        int prio = -1;

        int irq = -1;

        for (i = 63; i >= 0; --i) {

            if (flags & (1ULL<<i)) {

                int irq_prio = imx_avic_prio(s, i);

                if (irq_prio > prio) {

                    irq = i;

                    prio = irq_prio;

                }

            }

        }

        if (irq >= 0) {

            imx_avic_set_irq(s, irq, 0);

            return irq << 16 | prio;

        }

        return 0xffffffffULL;

    }

    case 17:/* Fast Interrupt vector and status register */

    {

        uint64_t flags = s->pending & s->enabled & s->is_fiq;

        int i = ctz64(flags);

        if (i < 64) {

            imx_avic_set_irq(opaque, i, 0);

            return i;

        }

        return 0xffffffffULL;

    }

    case 18:/* Interrupt source register high */

        return s->pending >> 32;



    case 19:/* Interrupt source register low */

        return s->pending & 0xffffffffULL;



    case 20:/* Interrupt Force Register high */

    case 21:/* Interrupt Force Register low */

        return 0;



    case 22:/* Normal Interrupt Pending Register High */

        return (s->pending & s->enabled & ~s->is_fiq) >> 32;



    case 23:/* Normal Interrupt Pending Register Low */

        return (s->pending & s->enabled & ~s->is_fiq) & 0xffffffffULL;



    case 24: /* Fast Interrupt Pending Register High  */

        return (s->pending & s->enabled & s->is_fiq) >> 32;



    case 25: /* Fast Interrupt Pending Register Low  */

        return (s->pending & s->enabled & s->is_fiq) & 0xffffffffULL;



    case 0x40:            /* AVIC vector 0, use for WFI WAR */

        return 0x4;



    default:

        IPRINTF("imx_avic_read: Bad offset 0x%x\n", (int)offset);

        return 0;

    }

}
