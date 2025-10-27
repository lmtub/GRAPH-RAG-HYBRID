void xen_be_unbind_evtchn(struct XenDevice *xendev)

{

    if (xendev->local_port == -1) {

        return;

    }

    qemu_set_fd_handler(xc_evtchn_fd(xendev->evtchndev), NULL, NULL, NULL);

    xc_evtchn_unbind(xendev->evtchndev, xendev->local_port);

    xen_be_printf(xendev, 2, "unbind evtchn port %d\n", xendev->local_port);

    xendev->local_port = -1;

}
