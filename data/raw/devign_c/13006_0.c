static void arm_timer_write(void *opaque, target_phys_addr_t offset,

                            uint32_t value)

{

    arm_timer_state *s = (arm_timer_state *)opaque;

    int freq;



    switch (offset >> 2) {

    case 0: /* TimerLoad */

        s->limit = value;

        arm_timer_recalibrate(s, 1);

        break;

    case 1: /* TimerValue */

        /* ??? Linux seems to want to write to this readonly register.

           Ignore it.  */

        break;

    case 2: /* TimerControl */

        if (s->control & TIMER_CTRL_ENABLE) {

            /* Pause the timer if it is running.  This may cause some

               inaccuracy dure to rounding, but avoids a whole lot of other

               messyness.  */

            ptimer_stop(s->timer);

        }

        s->control = value;

        freq = s->freq;

        /* ??? Need to recalculate expiry time after changing divisor.  */

        switch ((value >> 2) & 3) {

        case 1: freq >>= 4; break;

        case 2: freq >>= 8; break;

        }

        arm_timer_recalibrate(s, 0);

        ptimer_set_freq(s->timer, freq);

        if (s->control & TIMER_CTRL_ENABLE) {

            /* Restart the timer if still enabled.  */

            ptimer_run(s->timer, (s->control & TIMER_CTRL_ONESHOT) != 0);

        }

        break;

    case 3: /* TimerIntClr */

        s->int_level = 0;

        break;

    case 6: /* TimerBGLoad */

        s->limit = value;

        arm_timer_recalibrate(s, 0);

        break;

    default:

        hw_error("arm_timer_write: Bad offset %x\n", (int)offset);

    }

    arm_timer_update(s);

}
