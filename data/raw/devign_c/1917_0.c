void qemu_system_wakeup_request(WakeupReason reason)

{

    if (!is_suspended) {

        return;

    }

    if (!(wakeup_reason_mask & (1 << reason))) {

        return;

    }

    runstate_set(RUN_STATE_RUNNING);

    monitor_protocol_event(QEVENT_WAKEUP, NULL);

    notifier_list_notify(&wakeup_notifiers, &reason);

    reset_requested = 1;

    qemu_notify_event();

    is_suspended = false;

}
