void qemu_run_all_timers(void)

{

    /* rearm timer, if not periodic */

    if (alarm_timer->expired) {

        alarm_timer->expired = 0;

        qemu_rearm_alarm_timer(alarm_timer);

    }



    alarm_timer->pending = 0;



    /* vm time timers */

    if (vm_running) {

        qemu_run_timers(vm_clock);

    }



    qemu_run_timers(rt_clock);

    qemu_run_timers(host_clock);

}
