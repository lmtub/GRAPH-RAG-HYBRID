static void omap2_inth_write(void *opaque, target_phys_addr_t addr,

                             uint64_t value, unsigned size)

{

    struct omap_intr_handler_s *s = (struct omap_intr_handler_s *) opaque;

    int offset = addr;

    int bank_no, line_no;

    struct omap_intr_handler_bank_s *bank = NULL;



    if ((offset & 0xf80) == 0x80) {

        bank_no = (offset & 0x60) >> 5;

        if (bank_no < s->nbanks) {

            offset &= ~0x60;

            bank = &s->bank[bank_no];




        }

    }



    switch (offset) {

    case 0x10:	/* INTC_SYSCONFIG */

        s->autoidle &= 4;

        s->autoidle |= (value & 1) << 2;

        if (value & 2)						/* SOFTRESET */

            omap_inth_reset(&s->busdev.qdev);




    case 0x48:	/* INTC_CONTROL */

        s->mask = (value & 4) ? 0 : ~0;				/* GLOBALMASK */

        if (value & 2) {					/* NEWFIQAGR */

            qemu_set_irq(s->parent_intr[1], 0);

            s->new_agr[1] = ~0;

            omap_inth_update(s, 1);

        }

        if (value & 1) {					/* NEWIRQAGR */

            qemu_set_irq(s->parent_intr[0], 0);

            s->new_agr[0] = ~0;

            omap_inth_update(s, 0);

        }




    case 0x4c:	/* INTC_PROTECTION */

        /* TODO: Make a bitmap (or sizeof(char)map) of access privileges

         * for every register, see Chapter 3 and 4 for privileged mode.  */

        if (value & 1)

            fprintf(stderr, "%s: protection mode enable attempt\n",

                            __FUNCTION__);




    case 0x50:	/* INTC_IDLE */

        s->autoidle &= ~3;

        s->autoidle |= value & 3;




    /* Per-bank registers */

    case 0x84:	/* INTC_MIR */

        bank->mask = value;

        omap_inth_update(s, 0);

        omap_inth_update(s, 1);




    case 0x88:	/* INTC_MIR_CLEAR */

        bank->mask &= ~value;

        omap_inth_update(s, 0);

        omap_inth_update(s, 1);




    case 0x8c:	/* INTC_MIR_SET */

        bank->mask |= value;




    case 0x90:	/* INTC_ISR_SET */

        bank->irqs |= bank->swi |= value;

        omap_inth_update(s, 0);

        omap_inth_update(s, 1);




    case 0x94:	/* INTC_ISR_CLEAR */

        bank->swi &= ~value;

        bank->irqs = bank->swi & bank->inputs;




    /* Per-line registers */

    case 0x100 ... 0x300:	/* INTC_ILR */

        bank_no = (offset - 0x100) >> 7;

        if (bank_no > s->nbanks)

            break;

        bank = &s->bank[bank_no];

        line_no = (offset & 0x7f) >> 2;

        bank->priority[line_no] = (value >> 2) & 0x3f;

        bank->fiq &= ~(1 << line_no);

        bank->fiq |= (value & 1) << line_no;




    case 0x00:	/* INTC_REVISION */

    case 0x14:	/* INTC_SYSSTATUS */

    case 0x40:	/* INTC_SIR_IRQ */

    case 0x44:	/* INTC_SIR_FIQ */

    case 0x80:	/* INTC_ITR */

    case 0x98:	/* INTC_PENDING_IRQ */

    case 0x9c:	/* INTC_PENDING_FIQ */

        OMAP_RO_REG(addr);


    }


}