int xen_be_bind_evtchn(struct XenDevice *xendev)

{

    if (xendev->local_port != -1) {

        return 0;

    }

    xendev->local_port = xc_evtchn_bind_interdomain

        (xendev->evtchndev, xendev->dom, xendev->remote_port);

    if (xendev->local_port == -1) {

        xen_be_printf(xendev, 0, "xc_evtchn_bind_interdomain failed\n");

        return -1;

    }

    xen_be_printf(xendev, 2, "bind evtchn port %d\n", xendev->local_port);

    qemu_set_fd_handler(xc_evtchn_fd(xendev->evtchndev),

                        xen_be_evtchn_event, NULL, xendev);

    return 0;

}
