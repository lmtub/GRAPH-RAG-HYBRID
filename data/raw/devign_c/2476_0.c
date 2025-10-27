uint64_t ptimer_get_count(ptimer_state *s)

{

    int64_t now;

    uint64_t counter;



    if (s->enabled) {

        now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

        /* Figure out the current counter value.  */

        if (now - s->next_event > 0

            || s->period == 0) {

            /* Prevent timer underflowing if it should already have

               triggered.  */

            counter = 0;

        } else {

            uint64_t rem;

            uint64_t div;

            int clz1, clz2;

            int shift;



            /* We need to divide time by period, where time is stored in

               rem (64-bit integer) and period is stored in period/period_frac

               (64.32 fixed point).

              

               Doing full precision division is hard, so scale values and

               do a 64-bit division.  The result should be rounded down,

               so that the rounding error never causes the timer to go

               backwards.

            */



            rem = s->next_event - now;

            div = s->period;



            clz1 = clz64(rem);

            clz2 = clz64(div);

            shift = clz1 < clz2 ? clz1 : clz2;



            rem <<= shift;

            div <<= shift;

            if (shift >= 32) {

                div |= ((uint64_t)s->period_frac << (shift - 32));

            } else {

                if (shift != 0)

                    div |= (s->period_frac >> (32 - shift));

                /* Look at remaining bits of period_frac and round div up if 

                   necessary.  */

                if ((uint32_t)(s->period_frac << shift))

                    div += 1;

            }

            counter = rem / div;

        }

    } else {

        counter = s->delta;

    }

    return counter;

}
