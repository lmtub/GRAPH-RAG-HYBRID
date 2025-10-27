static target_ulong h_register_logical_lan(CPUPPCState *env,

                                           sPAPREnvironment *spapr,

                                           target_ulong opcode,

                                           target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong buf_list = args[1];

    target_ulong rec_queue = args[2];

    target_ulong filter_list = args[3];

    VIOsPAPRDevice *sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    VIOsPAPRVLANDevice *dev = (VIOsPAPRVLANDevice *)sdev;

    vlan_bd_t filter_list_bd;



    if (!dev) {

        return H_PARAMETER;

    }



    if (dev->isopen) {

        hcall_dprintf("H_REGISTER_LOGICAL_LAN called twice without "

                      "H_FREE_LOGICAL_LAN\n");

        return H_RESOURCE;

    }



    if (check_bd(dev, VLAN_VALID_BD(buf_list, SPAPR_VIO_TCE_PAGE_SIZE),

                 SPAPR_VIO_TCE_PAGE_SIZE) < 0) {

        hcall_dprintf("Bad buf_list 0x" TARGET_FMT_lx "\n", buf_list);

        return H_PARAMETER;

    }



    filter_list_bd = VLAN_VALID_BD(filter_list, SPAPR_VIO_TCE_PAGE_SIZE);

    if (check_bd(dev, filter_list_bd, SPAPR_VIO_TCE_PAGE_SIZE) < 0) {

        hcall_dprintf("Bad filter_list 0x" TARGET_FMT_lx "\n", filter_list);

        return H_PARAMETER;

    }



    if (!(rec_queue & VLAN_BD_VALID)

        || (check_bd(dev, rec_queue, VLAN_RQ_ALIGNMENT) < 0)) {

        hcall_dprintf("Bad receive queue\n");

        return H_PARAMETER;

    }



    dev->buf_list = buf_list;

    sdev->signal_state = 0;



    rec_queue &= ~VLAN_BD_TOGGLE;



    /* Initialize the buffer list */

    stq_tce(sdev, buf_list, rec_queue);

    stq_tce(sdev, buf_list + 8, filter_list_bd);

    spapr_tce_dma_zero(sdev, buf_list + VLAN_RX_BDS_OFF,

                       SPAPR_VIO_TCE_PAGE_SIZE - VLAN_RX_BDS_OFF);

    dev->add_buf_ptr = VLAN_RX_BDS_OFF - 8;

    dev->use_buf_ptr = VLAN_RX_BDS_OFF - 8;

    dev->rx_bufs = 0;

    dev->rxq_ptr = 0;



    /* Initialize the receive queue */

    spapr_tce_dma_zero(sdev, VLAN_BD_ADDR(rec_queue), VLAN_BD_LEN(rec_queue));



    dev->isopen = 1;

    return H_SUCCESS;

}
