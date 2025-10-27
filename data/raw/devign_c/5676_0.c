static int scsi_req_stream_length(SCSICommand *cmd, SCSIDevice *dev, uint8_t *buf)

{

    switch (buf[0]) {

    /* stream commands */

    case ERASE_12:

    case ERASE_16:

        cmd->xfer = 0;

        break;

    case READ_6:

    case READ_REVERSE:

    case RECOVER_BUFFERED_DATA:

    case WRITE_6:

        cmd->len = 6;

        cmd->xfer = buf[4] | (buf[3] << 8) | (buf[2] << 16);

        if (buf[1] & 0x01) { /* fixed */

            cmd->xfer *= dev->blocksize;

        }

        break;

    case READ_16:

    case READ_REVERSE_16:

    case VERIFY_16:

    case WRITE_16:

        cmd->len = 16;

        cmd->xfer = buf[14] | (buf[13] << 8) | (buf[12] << 16);

        if (buf[1] & 0x01) { /* fixed */

            cmd->xfer *= dev->blocksize;

        }

        break;

    case REWIND:

    case LOAD_UNLOAD:

        cmd->len = 6;

        cmd->xfer = 0;

        break;

    case SPACE_16:

        cmd->xfer = buf[13] | (buf[12] << 8);

        break;

    case READ_POSITION:

        cmd->xfer = buf[8] | (buf[7] << 8);

        break;

    case FORMAT_UNIT:

        cmd->xfer = buf[4] | (buf[3] << 8);

        break;

    /* generic commands */

    default:

        return scsi_req_length(cmd, dev, buf);

    }

    return 0;

}
