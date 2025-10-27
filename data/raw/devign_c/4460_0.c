static bool main_loop_should_exit(void)

{

    RunState r;

    ShutdownCause request;



    if (qemu_debug_requested()) {

        vm_stop(RUN_STATE_DEBUG);

    }

    if (qemu_suspend_requested()) {

        qemu_system_suspend();

    }

    request = qemu_shutdown_requested();

    if (request) {

        qemu_kill_report();

        /* TODO update event based on request */

        qapi_event_send_shutdown(&error_abort);

        if (no_shutdown) {

            vm_stop(RUN_STATE_SHUTDOWN);

        } else {

            return true;

        }

    }

    request = qemu_reset_requested();

    if (request) {

        pause_all_vcpus();

        qemu_system_reset(request);

        resume_all_vcpus();

        if (!runstate_check(RUN_STATE_RUNNING) &&

                !runstate_check(RUN_STATE_INMIGRATE)) {

            runstate_set(RUN_STATE_PRELAUNCH);

        }

    }

    if (qemu_wakeup_requested()) {

        pause_all_vcpus();

        qemu_system_reset(SHUTDOWN_CAUSE_NONE);

        notifier_list_notify(&wakeup_notifiers, &wakeup_reason);

        wakeup_reason = QEMU_WAKEUP_REASON_NONE;

        resume_all_vcpus();

        qapi_event_send_wakeup(&error_abort);

    }

    if (qemu_powerdown_requested()) {

        qemu_system_powerdown();

    }

    if (qemu_vmstop_requested(&r)) {

        vm_stop(r);

    }

    return false;

}
