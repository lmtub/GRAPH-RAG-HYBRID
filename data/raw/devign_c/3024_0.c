static void do_vm_stop(RunState state)

{

    if (runstate_is_running()) {

        cpu_disable_ticks();

        pause_all_vcpus();

        runstate_set(state);

        vm_state_notify(0, state);

        qemu_aio_flush();

        bdrv_flush_all();

        monitor_protocol_event(QEVENT_STOP, NULL);

    }

}
