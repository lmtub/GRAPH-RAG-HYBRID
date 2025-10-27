static int vm_request_pending(void)

{

    return powerdown_requested ||

           reset_requested ||

           shutdown_requested ||

           debug_requested ||

           vmstop_requested;

}
