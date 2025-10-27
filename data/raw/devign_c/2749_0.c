timer_write(void *opaque, target_phys_addr_t addr,

            uint64_t val64, unsigned int size)

{

    struct etrax_timer *t = opaque;

    uint32_t value = val64;



    switch (addr)

    {

        case RW_TMR0_DIV:

            t->rw_tmr0_div = value;

            break;

        case RW_TMR0_CTRL:

            D(printf ("RW_TMR0_CTRL=%x\n", value));

            t->rw_tmr0_ctrl = value;

            update_ctrl(t, 0);

            break;

        case RW_TMR1_DIV:

            t->rw_tmr1_div = value;

            break;

        case RW_TMR1_CTRL:

            D(printf ("RW_TMR1_CTRL=%x\n", value));

            t->rw_tmr1_ctrl = value;

            update_ctrl(t, 1);

            break;

        case RW_INTR_MASK:

            D(printf ("RW_INTR_MASK=%x\n", value));

            t->rw_intr_mask = value;

            timer_update_irq(t);

            break;

        case RW_WD_CTRL:

            timer_watchdog_update(t, value);

            break;

        case RW_ACK_INTR:

            t->rw_ack_intr = value;

            timer_update_irq(t);

            t->rw_ack_intr = 0;

            break;

        default:

            printf ("%s " TARGET_FMT_plx " %x\n",

                __func__, addr, value);

            break;

    }

}
