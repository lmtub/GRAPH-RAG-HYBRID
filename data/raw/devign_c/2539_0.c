static ssize_t net_rx_packet(NetClientState *nc, const uint8_t *buf, size_t size)

{

    struct XenNetDev *netdev = qemu_get_nic_opaque(nc);

    netif_rx_request_t rxreq;

    RING_IDX rc, rp;

    void *page;



    if (netdev->xendev.be_state != XenbusStateConnected) {

        return -1;

    }



    rc = netdev->rx_ring.req_cons;

    rp = netdev->rx_ring.sring->req_prod;

    xen_rmb(); /* Ensure we see queued requests up to 'rp'. */



    if (rc == rp || RING_REQUEST_CONS_OVERFLOW(&netdev->rx_ring, rc)) {

        xen_be_printf(&netdev->xendev, 2, "no buffer, drop packet\n");

        return -1;

    }

    if (size > XC_PAGE_SIZE - NET_IP_ALIGN) {

        xen_be_printf(&netdev->xendev, 0, "packet too big (%lu > %ld)",

                      (unsigned long)size, XC_PAGE_SIZE - NET_IP_ALIGN);

        return -1;

    }



    memcpy(&rxreq, RING_GET_REQUEST(&netdev->rx_ring, rc), sizeof(rxreq));

    netdev->rx_ring.req_cons = ++rc;



    page = xc_gnttab_map_grant_ref(netdev->xendev.gnttabdev,

                                   netdev->xendev.dom,

                                   rxreq.gref, PROT_WRITE);

    if (page == NULL) {

        xen_be_printf(&netdev->xendev, 0, "error: rx gref dereference failed (%d)\n",

                      rxreq.gref);

        net_rx_response(netdev, &rxreq, NETIF_RSP_ERROR, 0, 0, 0);

        return -1;

    }

    memcpy(page + NET_IP_ALIGN, buf, size);

    xc_gnttab_munmap(netdev->xendev.gnttabdev, page, 1);

    net_rx_response(netdev, &rxreq, NETIF_RSP_OKAY, NET_IP_ALIGN, size, 0);



    return size;

}
