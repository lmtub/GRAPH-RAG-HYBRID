static bool qemu_vmstop_requested(RunState *r)

{

    if (vmstop_requested < RUN_STATE_MAX) {

        *r = vmstop_requested;

        vmstop_requested = RUN_STATE_MAX;

        return true;

    }



    return false;

}
