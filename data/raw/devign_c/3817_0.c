static ssize_t spapr_vlan_receive(NetClientState *nc, const uint8_t *buf,

                                  size_t size)

{

    VIOsPAPRDevice *sdev = DO_UPCAST(NICState, nc, nc)->opaque;

    VIOsPAPRVLANDevice *dev = (VIOsPAPRVLANDevice *)sdev;

    vlan_bd_t rxq_bd = vio_ldq(sdev, dev->buf_list + VLAN_RXQ_BD_OFF);

    vlan_bd_t bd;

    int buf_ptr = dev->use_buf_ptr;

    uint64_t handle;

    uint8_t control;



    dprintf("spapr_vlan_receive() [%s] rx_bufs=%d\n", sdev->qdev.id,

            dev->rx_bufs);



    if (!dev->isopen) {

        return -1;

    }



    if (!dev->rx_bufs) {

        return -1;

    }



    do {

        buf_ptr += 8;

        if (buf_ptr >= SPAPR_TCE_PAGE_SIZE) {

            buf_ptr = VLAN_RX_BDS_OFF;

        }



        bd = vio_ldq(sdev, dev->buf_list + buf_ptr);

        dprintf("use_buf_ptr=%d bd=0x%016llx\n",

                buf_ptr, (unsigned long long)bd);

    } while ((!(bd & VLAN_BD_VALID) || (VLAN_BD_LEN(bd) < (size + 8)))

             && (buf_ptr != dev->use_buf_ptr));



    if (!(bd & VLAN_BD_VALID) || (VLAN_BD_LEN(bd) < (size + 8))) {

        /* Failed to find a suitable buffer */

        return -1;

    }



    /* Remove the buffer from the pool */

    dev->rx_bufs--;

    dev->use_buf_ptr = buf_ptr;

    vio_stq(sdev, dev->buf_list + dev->use_buf_ptr, 0);



    dprintf("Found buffer: ptr=%d num=%d\n", dev->use_buf_ptr, dev->rx_bufs);



    /* Transfer the packet data */

    if (spapr_vio_dma_write(sdev, VLAN_BD_ADDR(bd) + 8, buf, size) < 0) {

        return -1;

    }



    dprintf("spapr_vlan_receive: DMA write completed\n");



    /* Update the receive queue */

    control = VLAN_RXQC_TOGGLE | VLAN_RXQC_VALID;

    if (rxq_bd & VLAN_BD_TOGGLE) {

        control ^= VLAN_RXQC_TOGGLE;

    }



    handle = vio_ldq(sdev, VLAN_BD_ADDR(bd));

    vio_stq(sdev, VLAN_BD_ADDR(rxq_bd) + dev->rxq_ptr + 8, handle);

    vio_stl(sdev, VLAN_BD_ADDR(rxq_bd) + dev->rxq_ptr + 4, size);

    vio_sth(sdev, VLAN_BD_ADDR(rxq_bd) + dev->rxq_ptr + 2, 8);

    vio_stb(sdev, VLAN_BD_ADDR(rxq_bd) + dev->rxq_ptr, control);



    dprintf("wrote rxq entry (ptr=0x%llx): 0x%016llx 0x%016llx\n",

            (unsigned long long)dev->rxq_ptr,

            (unsigned long long)vio_ldq(sdev, VLAN_BD_ADDR(rxq_bd) +

                                        dev->rxq_ptr),

            (unsigned long long)vio_ldq(sdev, VLAN_BD_ADDR(rxq_bd) +

                                        dev->rxq_ptr + 8));



    dev->rxq_ptr += 16;

    if (dev->rxq_ptr >= VLAN_BD_LEN(rxq_bd)) {

        dev->rxq_ptr = 0;

        vio_stq(sdev, dev->buf_list + VLAN_RXQ_BD_OFF, rxq_bd ^ VLAN_BD_TOGGLE);

    }



    if (sdev->signal_state & 1) {

        qemu_irq_pulse(sdev->qirq);

    }



    return size;

}
