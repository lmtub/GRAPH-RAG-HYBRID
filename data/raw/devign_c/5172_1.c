static int net_connect(struct XenDevice *xendev)

{

    struct XenNetDev *netdev = container_of(xendev, struct XenNetDev, xendev);

    int rx_copy;



    if (xenstore_read_fe_int(&netdev->xendev, "tx-ring-ref",

                             &netdev->tx_ring_ref) == -1) {

        return -1;

    }

    if (xenstore_read_fe_int(&netdev->xendev, "rx-ring-ref",

                             &netdev->rx_ring_ref) == -1) {

        return 1;

    }

    if (xenstore_read_fe_int(&netdev->xendev, "event-channel",

                             &netdev->xendev.remote_port) == -1) {

        return -1;

    }



    if (xenstore_read_fe_int(&netdev->xendev, "request-rx-copy", &rx_copy) == -1) {

        rx_copy = 0;

    }

    if (rx_copy == 0) {

        xen_be_printf(&netdev->xendev, 0, "frontend doesn't support rx-copy.\n");

        return -1;

    }



    netdev->txs = xc_gnttab_map_grant_ref(netdev->xendev.gnttabdev,

                                          netdev->xendev.dom,

                                          netdev->tx_ring_ref,

                                          PROT_READ | PROT_WRITE);

    netdev->rxs = xc_gnttab_map_grant_ref(netdev->xendev.gnttabdev,

                                          netdev->xendev.dom,

                                          netdev->rx_ring_ref,

                                          PROT_READ | PROT_WRITE);

    if (!netdev->txs || !netdev->rxs) {

        return -1;

    }

    BACK_RING_INIT(&netdev->tx_ring, netdev->txs, XC_PAGE_SIZE);

    BACK_RING_INIT(&netdev->rx_ring, netdev->rxs, XC_PAGE_SIZE);



    xen_be_bind_evtchn(&netdev->xendev);



    xen_be_printf(&netdev->xendev, 1, "ok: tx-ring-ref %d, rx-ring-ref %d, "

                  "remote port %d, local port %d\n",

                  netdev->tx_ring_ref, netdev->rx_ring_ref,

                  netdev->xendev.remote_port, netdev->xendev.local_port);



    net_tx_packets(netdev);

    return 0;

}
