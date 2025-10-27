static void exynos4210_pwm_write(void *opaque, target_phys_addr_t offset,

        uint64_t value, unsigned size)

{

    Exynos4210PWMState *s = (Exynos4210PWMState *)opaque;

    int index;

    uint32_t new_val;

    int i;



    switch (offset) {

    case TCFG0: case TCFG1:

        index = (offset - TCFG0) >> 2;

        s->reg_tcfg[index] = value;



        /* update timers frequencies */

        for (i = 0; i < EXYNOS4210_PWM_TIMERS_NUM; i++) {

            exynos4210_pwm_update_freq(s, s->timer[i].id);

        }

        break;



    case TCON:

        for (i = 0; i < EXYNOS4210_PWM_TIMERS_NUM; i++) {

            if ((value & TCON_TIMER_MANUAL_UPD(i)) >

            (s->reg_tcon & TCON_TIMER_MANUAL_UPD(i))) {

                /*

                 * TCNTB and TCMPB are loaded into TCNT and TCMP.

                 * Update timers.

                 */



                /* this will start timer to run, this ok, because

                 * during processing start bit timer will be stopped

                 * if needed */

                ptimer_set_count(s->timer[i].ptimer, s->timer[i].reg_tcntb);

                DPRINTF("set timer %d count to %x\n", i,

                        s->timer[i].reg_tcntb);

            }



            if ((value & TCON_TIMER_START(i)) >

            (s->reg_tcon & TCON_TIMER_START(i))) {

                /* changed to start */

                ptimer_run(s->timer[i].ptimer, 1);

                DPRINTF("run timer %d\n", i);

            }



            if ((value & TCON_TIMER_START(i)) <

                    (s->reg_tcon & TCON_TIMER_START(i))) {

                /* changed to stop */

                ptimer_stop(s->timer[i].ptimer);

                DPRINTF("stop timer %d\n", i);

            }

        }

        s->reg_tcon = value;

        break;



    case TCNTB0: case TCNTB1:

    case TCNTB2: case TCNTB3: case TCNTB4:

        index = (offset - TCNTB0) / 0xC;

        s->timer[index].reg_tcntb = value;

        break;



    case TCMPB0: case TCMPB1:

    case TCMPB2: case TCMPB3:

        index = (offset - TCMPB0) / 0xC;

        s->timer[index].reg_tcmpb = value;

        break;



    case TINT_CSTAT:

        new_val = (s->reg_tint_cstat & 0x3E0) + (0x1F & value);

        new_val &= ~(0x3E0 & value);



        for (i = 0; i < EXYNOS4210_PWM_TIMERS_NUM; i++) {

            if ((new_val & TINT_CSTAT_STATUS(i)) <

                    (s->reg_tint_cstat & TINT_CSTAT_STATUS(i))) {

                qemu_irq_lower(s->timer[i].irq);

            }

        }



        s->reg_tint_cstat = new_val;

        break;



    default:

        fprintf(stderr,

                "[exynos4210.pwm: bad write offset " TARGET_FMT_plx "]\n",

                offset);

        break;



    }

}
