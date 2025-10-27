void qemu_system_shutdown_request(void)

{

    trace_qemu_system_shutdown_request();

    replay_shutdown_request();

    shutdown_requested = 1;

    qemu_notify_event();

}
