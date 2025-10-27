static int vscsi_send_iu(VSCSIState *s, vscsi_req *req,

                         uint64_t length, uint8_t format)

{

    long rc, rc1;



    /* First copy the SRP */

    rc = spapr_vio_dma_write(&s->vdev, req->crq.s.IU_data_ptr,

                             &req->iu, length);

    if (rc) {

        fprintf(stderr, "vscsi_send_iu: DMA write failure !\n");

    }



    req->crq.s.valid = 0x80;

    req->crq.s.format = format;

    req->crq.s.reserved = 0x00;

    req->crq.s.timeout = cpu_to_be16(0x0000);

    req->crq.s.IU_length = cpu_to_be16(length);

    req->crq.s.IU_data_ptr = req->iu.srp.rsp.tag; /* right byte order */



    if (rc == 0) {

        req->crq.s.status = 0x99; /* Just needs to be non-zero */

    } else {

        req->crq.s.status = 0x00;

    }



    rc1 = spapr_vio_send_crq(&s->vdev, req->crq.raw);

    if (rc1) {

        fprintf(stderr, "vscsi_send_iu: Error sending response\n");

        return rc1;

    }



    return rc;

}
