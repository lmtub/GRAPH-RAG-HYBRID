void qemu_system_reset_request(void)

{

    if (no_reboot) {

        shutdown_requested = 1;

    } else {

        reset_requested = 1;

    }

    cpu_stop_current();

    qemu_notify_event();

}
