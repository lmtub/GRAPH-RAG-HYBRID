void qemu_system_guest_panicked(void)

{




    qapi_event_send_guest_panicked(GUEST_PANIC_ACTION_PAUSE, &error_abort);

    vm_stop(RUN_STATE_GUEST_PANICKED);
