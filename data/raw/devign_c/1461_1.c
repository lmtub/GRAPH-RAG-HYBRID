static void gic_dist_writeb(void *opaque, hwaddr offset,

                            uint32_t value, MemTxAttrs attrs)

{

    GICState *s = (GICState *)opaque;

    int irq;

    int i;

    int cpu;



    cpu = gic_get_current_cpu(s);

    if (offset < 0x100) {

        if (offset == 0) {

            if (s->security_extn && !attrs.secure) {

                /* NS version is just an alias of the S version's bit 1 */

                s->ctlr = deposit32(s->ctlr, 1, 1, value);

            } else if (gic_has_groups(s)) {

                s->ctlr = value & (GICD_CTLR_EN_GRP0 | GICD_CTLR_EN_GRP1);

            } else {

                s->ctlr = value & GICD_CTLR_EN_GRP0;

            }

            DPRINTF("Distributor: Group0 %sabled; Group 1 %sabled\n",

                    s->ctlr & GICD_CTLR_EN_GRP0 ? "En" : "Dis",

                    s->ctlr & GICD_CTLR_EN_GRP1 ? "En" : "Dis");

        } else if (offset < 4) {

            /* ignored.  */

        } else if (offset >= 0x80) {

            /* Interrupt Group Registers: RAZ/WI for NS access to secure

             * GIC, or for GICs without groups.

             */

            if (!(s->security_extn && !attrs.secure) && gic_has_groups(s)) {

                /* Every byte offset holds 8 group status bits */

                irq = (offset - 0x80) * 8 + GIC_BASE_IRQ;

                if (irq >= s->num_irq) {

                    goto bad_reg;

                }

                for (i = 0; i < 8; i++) {

                    /* Group bits are banked for private interrupts */

                    int cm = (irq < GIC_INTERNAL) ? (1 << cpu) : ALL_CPU_MASK;

                    if (value & (1 << i)) {

                        /* Group1 (Non-secure) */

                        GIC_SET_GROUP(irq + i, cm);

                    } else {

                        /* Group0 (Secure) */

                        GIC_CLEAR_GROUP(irq + i, cm);

                    }

                }

            }

        } else {

            goto bad_reg;

        }

    } else if (offset < 0x180) {

        /* Interrupt Set Enable.  */

        irq = (offset - 0x100) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_NR_SGIS) {

            value = 0xff;

        }



        for (i = 0; i < 8; i++) {

            if (value & (1 << i)) {

                int mask =

                    (irq < GIC_INTERNAL) ? (1 << cpu) : GIC_TARGET(irq + i);

                int cm = (irq < GIC_INTERNAL) ? (1 << cpu) : ALL_CPU_MASK;



                if (!GIC_TEST_ENABLED(irq + i, cm)) {

                    DPRINTF("Enabled IRQ %d\n", irq + i);


                }

                GIC_SET_ENABLED(irq + i, cm);

                /* If a raised level triggered IRQ enabled then mark

                   is as pending.  */

                if (GIC_TEST_LEVEL(irq + i, mask)

                        && !GIC_TEST_EDGE_TRIGGER(irq + i)) {

                    DPRINTF("Set %d pending mask %x\n", irq + i, mask);

                    GIC_SET_PENDING(irq + i, mask);

                }

            }

        }

    } else if (offset < 0x200) {

        /* Interrupt Clear Enable.  */

        irq = (offset - 0x180) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_NR_SGIS) {

            value = 0;

        }



        for (i = 0; i < 8; i++) {

            if (value & (1 << i)) {

                int cm = (irq < GIC_INTERNAL) ? (1 << cpu) : ALL_CPU_MASK;



                if (GIC_TEST_ENABLED(irq + i, cm)) {

                    DPRINTF("Disabled IRQ %d\n", irq + i);

                    trace_gic_disable_irq(irq + i);

                }

                GIC_CLEAR_ENABLED(irq + i, cm);

            }

        }

    } else if (offset < 0x280) {

        /* Interrupt Set Pending.  */

        irq = (offset - 0x200) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_NR_SGIS) {

            value = 0;

        }



        for (i = 0; i < 8; i++) {

            if (value & (1 << i)) {

                GIC_SET_PENDING(irq + i, GIC_TARGET(irq + i));

            }

        }

    } else if (offset < 0x300) {

        /* Interrupt Clear Pending.  */

        irq = (offset - 0x280) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_NR_SGIS) {

            value = 0;

        }



        for (i = 0; i < 8; i++) {

            /* ??? This currently clears the pending bit for all CPUs, even

               for per-CPU interrupts.  It's unclear whether this is the

               corect behavior.  */

            if (value & (1 << i)) {

                GIC_CLEAR_PENDING(irq + i, ALL_CPU_MASK);

            }

        }

    } else if (offset < 0x400) {

        /* Interrupt Active.  */

        goto bad_reg;

    } else if (offset < 0x800) {

        /* Interrupt Priority.  */

        irq = (offset - 0x400) + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        gic_set_priority(s, cpu, irq, value, attrs);

    } else if (offset < 0xc00) {

        /* Interrupt CPU Target. RAZ/WI on uniprocessor GICs, with the

         * annoying exception of the 11MPCore's GIC.

         */

        if (s->num_cpu != 1 || s->revision == REV_11MPCORE) {

            irq = (offset - 0x800) + GIC_BASE_IRQ;

            if (irq >= s->num_irq) {

                goto bad_reg;

            }

            if (irq < 29) {

                value = 0;

            } else if (irq < GIC_INTERNAL) {

                value = ALL_CPU_MASK;

            }

            s->irq_target[irq] = value & ALL_CPU_MASK;

        }

    } else if (offset < 0xf00) {

        /* Interrupt Configuration.  */

        irq = (offset - 0xc00) * 4 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_NR_SGIS)

            value |= 0xaa;

        for (i = 0; i < 4; i++) {

            if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

                if (value & (1 << (i * 2))) {

                    GIC_SET_MODEL(irq + i);

                } else {

                    GIC_CLEAR_MODEL(irq + i);

                }

            }

            if (value & (2 << (i * 2))) {

                GIC_SET_EDGE_TRIGGER(irq + i);

            } else {

                GIC_CLEAR_EDGE_TRIGGER(irq + i);

            }

        }

    } else if (offset < 0xf10) {

        /* 0xf00 is only handled for 32-bit writes.  */

        goto bad_reg;

    } else if (offset < 0xf20) {

        /* GICD_CPENDSGIRn */

        if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

            goto bad_reg;

        }

        irq = (offset - 0xf10);



        s->sgi_pending[irq][cpu] &= ~value;

        if (s->sgi_pending[irq][cpu] == 0) {

            GIC_CLEAR_PENDING(irq, 1 << cpu);

        }

    } else if (offset < 0xf30) {

        /* GICD_SPENDSGIRn */

        if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

            goto bad_reg;

        }

        irq = (offset - 0xf20);



        GIC_SET_PENDING(irq, 1 << cpu);

        s->sgi_pending[irq][cpu] |= value;

    } else {

        goto bad_reg;

    }

    gic_update(s);

    return;

bad_reg:

    qemu_log_mask(LOG_GUEST_ERROR,

                  "gic_dist_writeb: Bad offset %x\n", (int)offset);

}