static int scsi_req_length(SCSIRequest *req, uint8_t *cmd)

{

    switch (cmd[0] >> 5) {

    case 0:

        req->cmd.xfer = cmd[4];

        req->cmd.len = 6;

        /* length 0 means 256 blocks */

        if (req->cmd.xfer == 0)

            req->cmd.xfer = 256;

        break;

    case 1:

    case 2:

        req->cmd.xfer = cmd[8] | (cmd[7] << 8);

        req->cmd.len = 10;

        break;

    case 4:

        req->cmd.xfer = cmd[13] | (cmd[12] << 8) | (cmd[11] << 16) | (cmd[10] << 24);

        req->cmd.len = 16;

        break;

    case 5:

        req->cmd.xfer = cmd[9] | (cmd[8] << 8) | (cmd[7] << 16) | (cmd[6] << 24);

        req->cmd.len = 12;

        break;

    default:

        trace_scsi_req_parse_bad(req->dev->id, req->lun, req->tag, cmd[0]);

        return -1;

    }



    switch(cmd[0]) {

    case TEST_UNIT_READY:

    case START_STOP:

    case SEEK_6:

    case WRITE_FILEMARKS:

    case SPACE:

    case RESERVE:

    case RELEASE:

    case ERASE:

    case ALLOW_MEDIUM_REMOVAL:

    case VERIFY:

    case SEEK_10:

    case SYNCHRONIZE_CACHE:

    case LOCK_UNLOCK_CACHE:

    case LOAD_UNLOAD:

    case SET_CD_SPEED:

    case SET_LIMITS:

    case WRITE_LONG:

    case MOVE_MEDIUM:

    case UPDATE_BLOCK:

        req->cmd.xfer = 0;

        break;

    case MODE_SENSE:

        break;

    case WRITE_SAME:

        req->cmd.xfer = 1;

        break;

    case READ_CAPACITY:

        req->cmd.xfer = 8;

        break;

    case READ_BLOCK_LIMITS:

        req->cmd.xfer = 6;

        break;

    case READ_POSITION:

        req->cmd.xfer = 20;

        break;

    case SEND_VOLUME_TAG:

        req->cmd.xfer *= 40;

        break;

    case MEDIUM_SCAN:

        req->cmd.xfer *= 8;

        break;

    case WRITE_10:

    case WRITE_VERIFY:

    case WRITE_6:

    case WRITE_12:

    case WRITE_VERIFY_12:

    case WRITE_16:

    case WRITE_VERIFY_16:

        req->cmd.xfer *= req->dev->blocksize;

        break;

    case READ_10:

    case READ_6:

    case READ_REVERSE:

    case RECOVER_BUFFERED_DATA:

    case READ_12:

    case READ_16:

        req->cmd.xfer *= req->dev->blocksize;

        break;

    case INQUIRY:

        req->cmd.xfer = cmd[4] | (cmd[3] << 8);

        break;

    case MAINTENANCE_OUT:

    case MAINTENANCE_IN:

        if (req->dev->type == TYPE_ROM) {

            /* GPCMD_REPORT_KEY and GPCMD_SEND_KEY from multi media commands */

            req->cmd.xfer = cmd[9] | (cmd[8] << 8);

        }

        break;

    }

    return 0;

}
