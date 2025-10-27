static int vscsi_send_adapter_info(VSCSIState *s, vscsi_req *req)

{

    struct viosrp_adapter_info *sinfo;

    struct mad_adapter_info_data info;

    int rc;



    sinfo = &req->iu.mad.adapter_info;



#if 0 /* What for ? */

    rc = spapr_tce_dma_read(&s->vdev, be64_to_cpu(sinfo->buffer),

                            &info, be16_to_cpu(sinfo->common.length));

    if (rc) {

        fprintf(stderr, "vscsi_send_adapter_info: DMA read failure !\n");

    }

#endif

    memset(&info, 0, sizeof(info));

    strcpy(info.srp_version, SRP_VERSION);

    strncpy(info.partition_name, "qemu", sizeof("qemu"));

    info.partition_number = cpu_to_be32(0);

    info.mad_version = cpu_to_be32(1);

    info.os_type = cpu_to_be32(2);

    info.port_max_txu[0] = cpu_to_be32(VSCSI_MAX_SECTORS << 9);



    rc = spapr_tce_dma_write(&s->vdev, be64_to_cpu(sinfo->buffer),

                             &info, be16_to_cpu(sinfo->common.length));

    if (rc)  {

        fprintf(stderr, "vscsi_send_adapter_info: DMA write failure !\n");

    }



    sinfo->common.status = rc ? cpu_to_be32(1) : 0;



    return vscsi_send_iu(s, req, sizeof(*sinfo), VIOSRP_MAD_FORMAT);

}
