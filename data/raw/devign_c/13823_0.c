static void imx_timerg_write(void *opaque, target_phys_addr_t offset,

                             uint64_t value, unsigned size)

{

    IMXTimerGState *s = (IMXTimerGState *)opaque;

    DPRINTF("g-write(offset=%x, value = 0x%x)\n", (unsigned int)offset >> 2,

            (unsigned int)value);



    switch (offset >> 2) {

    case 0: {

        uint32_t oldcr = s->cr;

        /* CR */

        if (value & GPT_CR_SWR) { /* force reset */

            value &= ~GPT_CR_SWR;

            imx_timerg_reset(&s->busdev.qdev);

            imx_timerg_update(s);

        }



        s->cr = value & ~0x7c00;

        imx_timerg_set_freq(s);

        if ((oldcr ^ value) & GPT_CR_EN) {

            if (value & GPT_CR_EN) {

                if (value & GPT_CR_ENMOD) {

                    ptimer_set_count(s->timer, s->ocr1);

                    s->cnt = 0;

                }

                ptimer_run(s->timer,

                           (value & GPT_CR_FRR) && (s->ocr1 != TIMER_MAX));

            } else {

                ptimer_stop(s->timer);

            };

        }

        return;

    }



    case 1: /* Prescaler */

        s->pr = value & 0xfff;

        imx_timerg_set_freq(s);

        return;



    case 2: /* SR */

        /*

         * No point in implementing the status register bits to do with

         * external interrupt sources.

         */

        value &= GPT_SR_OF1 | GPT_SR_ROV;

        s->sr &= ~value;

        imx_timerg_update(s);

        return;



    case 3: /* IR -- interrupt register */

        s->ir = value & 0x3f;

        imx_timerg_update(s);

        return;



    case 4: /* OCR1 -- output compare register */

        /* In non-freerun mode, reset count when this register is written */

        if (!(s->cr & GPT_CR_FRR)) {

            s->waiting_rov = 0;

            ptimer_set_limit(s->timer, value, 1);

        } else {

            imx_timerg_update_counts(s);

            if (value > s->cnt) {

                s->waiting_rov = 0;

                imx_timerg_reload(s, value);

            } else {

                s->waiting_rov = 1;

                imx_timerg_reload(s, TIMER_MAX - s->cnt);

            }

        }

        s->ocr1 = value;

        return;



    default:

        IPRINTF("imx_timerg_write: Bad offset %x\n",

                (int)offset >> 2);

    }

}
