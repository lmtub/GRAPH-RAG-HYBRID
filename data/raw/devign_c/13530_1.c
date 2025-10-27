void qemu_system_vmstop_request(RunState state)

{

    vmstop_requested = state;

    qemu_notify_event();

}
