int64_t qemu_get_clock(QEMUClock *clock)

{

    switch(clock->type) {

    case QEMU_TIMER_REALTIME:

        return get_clock() / 1000000;

    default:

    case QEMU_TIMER_VIRTUAL:

        if (use_icount) {

            return cpu_get_icount();

        } else {

            return cpu_get_clock();

        }

    }

}
