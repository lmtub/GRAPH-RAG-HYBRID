static target_ulong h_add_logical_lan_buffer(PowerPCCPU *cpu,

                                             sPAPRMachineState *spapr,

                                             target_ulong opcode,

                                             target_ulong *args)

{

    target_ulong reg = args[0];

    target_ulong buf = args[1];

    VIOsPAPRDevice *sdev = spapr_vio_find_by_reg(spapr->vio_bus, reg);

    VIOsPAPRVLANDevice *dev = VIO_SPAPR_VLAN_DEVICE(sdev);

    target_long ret;



    DPRINTF("H_ADD_LOGICAL_LAN_BUFFER(0x" TARGET_FMT_lx

            ", 0x" TARGET_FMT_lx ")\n", reg, buf);



    if (!sdev) {

        hcall_dprintf("Bad device\n");

        return H_PARAMETER;

    }



    if ((check_bd(dev, buf, 4) < 0)

        || (VLAN_BD_LEN(buf) < 16)) {

        hcall_dprintf("Bad buffer enqueued\n");

        return H_PARAMETER;

    }



    if (!dev->isopen) {

        return H_RESOURCE;

    }



    ret = spapr_vlan_add_rxbuf_to_page(dev, buf);

    if (ret) {

        return ret;

    }



    dev->rx_bufs++;



    qemu_flush_queued_packets(qemu_get_queue(dev->nic));



    return H_SUCCESS;

}
