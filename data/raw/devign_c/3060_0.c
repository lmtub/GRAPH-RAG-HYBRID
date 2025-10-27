int virtio_blk_handle_scsi_req(VirtIOBlock *blk,

                               VirtQueueElement *elem)

{

    int status = VIRTIO_BLK_S_OK;

    struct virtio_scsi_inhdr *scsi = NULL;

    VirtIODevice *vdev = VIRTIO_DEVICE(blk);



#ifdef __linux__

    int i;

    struct sg_io_hdr hdr;

#endif



    /*

     * We require at least one output segment each for the virtio_blk_outhdr

     * and the SCSI command block.

     *

     * We also at least require the virtio_blk_inhdr, the virtio_scsi_inhdr

     * and the sense buffer pointer in the input segments.

     */

    if (elem->out_num < 2 || elem->in_num < 3) {

        status = VIRTIO_BLK_S_IOERR;

        goto fail;

    }



    /*

     * The scsi inhdr is placed in the second-to-last input segment, just

     * before the regular inhdr.

     */

    scsi = (void *)elem->in_sg[elem->in_num - 2].iov_base;



    if (!blk->conf.scsi) {

        status = VIRTIO_BLK_S_UNSUPP;

        goto fail;

    }



    /*

     * No support for bidirection commands yet.

     */

    if (elem->out_num > 2 && elem->in_num > 3) {

        status = VIRTIO_BLK_S_UNSUPP;

        goto fail;

    }



#ifdef __linux__

    memset(&hdr, 0, sizeof(struct sg_io_hdr));

    hdr.interface_id = 'S';

    hdr.cmd_len = elem->out_sg[1].iov_len;

    hdr.cmdp = elem->out_sg[1].iov_base;

    hdr.dxfer_len = 0;



    if (elem->out_num > 2) {

        /*

         * If there are more than the minimally required 2 output segments

         * there is write payload starting from the third iovec.

         */

        hdr.dxfer_direction = SG_DXFER_TO_DEV;

        hdr.iovec_count = elem->out_num - 2;



        for (i = 0; i < hdr.iovec_count; i++)

            hdr.dxfer_len += elem->out_sg[i + 2].iov_len;



        hdr.dxferp = elem->out_sg + 2;



    } else if (elem->in_num > 3) {

        /*

         * If we have more than 3 input segments the guest wants to actually

         * read data.

         */

        hdr.dxfer_direction = SG_DXFER_FROM_DEV;

        hdr.iovec_count = elem->in_num - 3;

        for (i = 0; i < hdr.iovec_count; i++)

            hdr.dxfer_len += elem->in_sg[i].iov_len;



        hdr.dxferp = elem->in_sg;

    } else {

        /*

         * Some SCSI commands don't actually transfer any data.

         */

        hdr.dxfer_direction = SG_DXFER_NONE;

    }



    hdr.sbp = elem->in_sg[elem->in_num - 3].iov_base;

    hdr.mx_sb_len = elem->in_sg[elem->in_num - 3].iov_len;



    status = bdrv_ioctl(blk->bs, SG_IO, &hdr);

    if (status) {

        status = VIRTIO_BLK_S_UNSUPP;

        goto fail;

    }



    /*

     * From SCSI-Generic-HOWTO: "Some lower level drivers (e.g. ide-scsi)

     * clear the masked_status field [hence status gets cleared too, see

     * block/scsi_ioctl.c] even when a CHECK_CONDITION or COMMAND_TERMINATED

     * status has occurred.  However they do set DRIVER_SENSE in driver_status

     * field. Also a (sb_len_wr > 0) indicates there is a sense buffer.

     */

    if (hdr.status == 0 && hdr.sb_len_wr > 0) {

        hdr.status = CHECK_CONDITION;

    }



    virtio_stl_p(vdev, &scsi->errors,

                 hdr.status | (hdr.msg_status << 8) |

                 (hdr.host_status << 16) | (hdr.driver_status << 24));

    virtio_stl_p(vdev, &scsi->residual, hdr.resid);

    virtio_stl_p(vdev, &scsi->sense_len, hdr.sb_len_wr);

    virtio_stl_p(vdev, &scsi->data_len, hdr.dxfer_len);



    return status;

#else

    abort();

#endif



fail:

    /* Just put anything nonzero so that the ioctl fails in the guest.  */

    if (scsi) {

        virtio_stl_p(vdev, &scsi->errors, 255);

    }

    return status;

}
