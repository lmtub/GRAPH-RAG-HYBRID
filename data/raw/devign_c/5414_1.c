static bool scsi_block_is_passthrough(SCSIDiskState *s, uint8_t *buf)

{

    switch (buf[0]) {

    case VERIFY_10:

    case VERIFY_12:

    case VERIFY_16:

        /* Check if BYTCHK == 0x01 (data-out buffer contains data

         * for the number of logical blocks specified in the length

         * field).  For other modes, do not use scatter/gather operation.

         */

        if ((buf[1] & 6) != 2) {

            return false;

        }

        break;



    case READ_6:

    case READ_10:

    case READ_12:

    case READ_16:

    case WRITE_6:

    case WRITE_10:

    case WRITE_12:

    case WRITE_16:

    case WRITE_VERIFY_10:

    case WRITE_VERIFY_12:

    case WRITE_VERIFY_16:

        /* MMC writing cannot be done via DMA helpers, because it sometimes

         * involves writing beyond the maximum LBA or to negative LBA (lead-in).

         * We might use scsi_disk_dma_reqops as long as no writing commands are

         * seen, but performance usually isn't paramount on optical media.  So,

         * just make scsi-block operate the same as scsi-generic for them.

         */

        if (s->qdev.type != TYPE_ROM) {

            return false;

        }

        break;



    default:

        break;

    }



    return true;

}
