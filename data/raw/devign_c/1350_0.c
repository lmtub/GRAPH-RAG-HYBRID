void vm_start(void)

{

    if (!vm_running) {

        cpu_enable_ticks();

        vm_running = 1;

        vm_state_notify(1, 0);

        qemu_rearm_alarm_timer(alarm_timer);

        resume_all_vcpus();

    }

}
