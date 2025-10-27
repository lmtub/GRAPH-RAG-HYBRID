void qemu_system_debug_request(void)

{

    debug_requested = 1;

    vm_stop(VMSTOP_DEBUG);

}
