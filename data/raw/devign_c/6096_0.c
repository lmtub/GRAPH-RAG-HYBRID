static int megasas_dcmd_ld_get_list(MegasasState *s, MegasasCmd *cmd)

{

    struct mfi_ld_list info;

    size_t dcmd_size = sizeof(info), resid;

    uint32_t num_ld_disks = 0, max_ld_disks = s->fw_luns;

    uint64_t ld_size;

    BusChild *kid;



    memset(&info, 0, dcmd_size);

    if (cmd->iov_size < dcmd_size) {

        trace_megasas_dcmd_invalid_xfer_len(cmd->index, cmd->iov_size,

                                            dcmd_size);

        return MFI_STAT_INVALID_PARAMETER;

    }



    if (megasas_is_jbod(s)) {

        max_ld_disks = 0;

    }

    QTAILQ_FOREACH(kid, &s->bus.qbus.children, sibling) {

        SCSIDevice *sdev = DO_UPCAST(SCSIDevice, qdev, kid->child);

        BlockConf *conf = &sdev->conf;



        if (num_ld_disks >= max_ld_disks) {

            break;

        }

        /* Logical device size is in blocks */

        bdrv_get_geometry(conf->bs, &ld_size);

        info.ld_list[num_ld_disks].ld.v.target_id = sdev->id;

        info.ld_list[num_ld_disks].ld.v.lun_id = sdev->lun;

        info.ld_list[num_ld_disks].state = MFI_LD_STATE_OPTIMAL;

        info.ld_list[num_ld_disks].size = cpu_to_le64(ld_size);

        num_ld_disks++;

    }

    info.ld_count = cpu_to_le32(num_ld_disks);

    trace_megasas_dcmd_ld_get_list(cmd->index, num_ld_disks, max_ld_disks);



    resid = dma_buf_read((uint8_t *)&info, dcmd_size, &cmd->qsg);

    cmd->iov_size = dcmd_size - resid;

    return MFI_STAT_OK;

}
