static void timerlist_rearm(QEMUTimerList *timer_list)

{

    /* Interrupt execution to force deadline recalculation.  */

    if (timer_list->clock->type == QEMU_CLOCK_VIRTUAL) {

        qemu_start_warp_timer();

    }

    timerlist_notify(timer_list);

}
