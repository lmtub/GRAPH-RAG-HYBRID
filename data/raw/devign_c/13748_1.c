static void vscsi_command_complete(SCSIBus *bus, int reason, uint32_t tag,

                                   uint32_t arg)

{

    VSCSIState *s = DO_UPCAST(VSCSIState, vdev.qdev, bus->qbus.parent);

    vscsi_req *req = vscsi_find_req(s, tag);

    SCSIDevice *sdev;

    uint8_t *buf;

    int32_t res_in = 0, res_out = 0;

    int len, rc = 0;



    dprintf("VSCSI: SCSI cmd complete, r=0x%x tag=0x%x arg=0x%x, req=%p\n",

            reason, tag, arg, req);

    if (req == NULL) {

        fprintf(stderr, "VSCSI: Can't find request for tag 0x%x\n", tag);

        return;

    }

    sdev = req->sdev;



    if (req->sensing) {

        if (reason == SCSI_REASON_DONE) {

            dprintf("VSCSI: Sense done !\n");

            vscsi_send_rsp(s, req, CHECK_CONDITION, 0, 0);

            vscsi_put_req(s, req);

        } else {

            uint8_t *buf = sdev->info->get_buf(sdev, tag);



            len = MIN(arg, SCSI_SENSE_BUF_SIZE);

            dprintf("VSCSI: Sense data, %d bytes:\n", len);

            dprintf("       %02x  %02x  %02x  %02x  %02x  %02x  %02x  %02x\n",

                    buf[0], buf[1], buf[2], buf[3],

                    buf[4], buf[5], buf[6], buf[7]);

            dprintf("       %02x  %02x  %02x  %02x  %02x  %02x  %02x  %02x\n",

                    buf[8], buf[9], buf[10], buf[11],

                    buf[12], buf[13], buf[14], buf[15]);

            memcpy(req->sense, buf, len);

            req->senselen = len;

            sdev->info->read_data(sdev, req->qtag);

        }

        return;

    }



    if (reason == SCSI_REASON_DONE) {

        dprintf("VSCSI: Command complete err=%d\n", arg);

        if (arg == 0) {

            /* We handle overflows, not underflows for normal commands,

             * but hopefully nobody cares

             */

            if (req->writing) {

                res_out = req->data_len;

            } else {

                res_in = req->data_len;

            }

            vscsi_send_rsp(s, req, 0, res_in, res_out);

        } else if (arg == CHECK_CONDITION) {

            dprintf("VSCSI: Got CHECK_CONDITION, requesting sense...\n");

            vscsi_send_request_sense(s, req);

            return;

        } else {

            vscsi_send_rsp(s, req, arg, 0, 0);

        }

        vscsi_put_req(s, req);

        return;

    }



    /* "arg" is how much we have read for reads and how much we want

     * to write for writes (ie, how much is to be DMA'd)

     */

    if (arg) {

        buf = sdev->info->get_buf(sdev, tag);

        rc = vscsi_srp_transfer_data(s, req, req->writing, buf, arg);

    }

    if (rc < 0) {

        fprintf(stderr, "VSCSI: RDMA error rc=%d!\n", rc);

        sdev->info->cancel_io(sdev, req->qtag);

        vscsi_makeup_sense(s, req, HARDWARE_ERROR, 0, 0);

        vscsi_send_rsp(s, req, CHECK_CONDITION, 0, 0);

        vscsi_put_req(s, req);

        return;

    }



    /* Start next chunk */

    req->data_len -= rc;

    if (req->writing) {

        sdev->info->write_data(sdev, req->qtag);

    } else {

        sdev->info->read_data(sdev, req->qtag);

    }

}
