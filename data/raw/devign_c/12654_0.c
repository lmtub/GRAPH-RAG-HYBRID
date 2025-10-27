static void scsi_read_complete(void * opaque, int ret)

{

    SCSIGenericReq *r = (SCSIGenericReq *)opaque;

    SCSIDevice *s = r->req.dev;

    int len;



    r->req.aiocb = NULL;

    if (ret || r->req.io_canceled) {

        scsi_command_complete(r, ret);

        return;

    }

    len = r->io_header.dxfer_len - r->io_header.resid;

    DPRINTF("Data ready tag=0x%x len=%d\n", r->req.tag, len);



    r->len = -1;

    if (len == 0) {

        scsi_command_complete(r, 0);

    } else {

        /* Snoop READ CAPACITY output to set the blocksize.  */

        if (r->req.cmd.buf[0] == READ_CAPACITY_10 &&

            (ldl_be_p(&r->buf[0]) != 0xffffffffU || s->max_lba == 0)) {

            s->blocksize = ldl_be_p(&r->buf[4]);

            s->max_lba = ldl_be_p(&r->buf[0]) & 0xffffffffULL;

        } else if (r->req.cmd.buf[0] == SERVICE_ACTION_IN_16 &&

                   (r->req.cmd.buf[1] & 31) == SAI_READ_CAPACITY_16) {

            s->blocksize = ldl_be_p(&r->buf[8]);

            s->max_lba = ldq_be_p(&r->buf[0]);

        }

        bdrv_set_guest_block_size(s->conf.bs, s->blocksize);



        scsi_req_data(&r->req, len);

        scsi_req_unref(&r->req);

    }

}
