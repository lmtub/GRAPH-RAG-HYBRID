static void xen_hvm_change_state_handler(void *opaque, int running,

                                         RunState rstate)

{

    if (running) {

        xen_main_loop_prepare((XenIOState *)opaque);

    }

}
