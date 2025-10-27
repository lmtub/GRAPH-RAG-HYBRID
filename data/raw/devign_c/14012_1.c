static void scsi_cmd_xfer_mode(SCSICommand *cmd)

{

    if (!cmd->xfer) {

        cmd->mode = SCSI_XFER_NONE;

        return;

    }

    switch (cmd->buf[0]) {

    case WRITE_6:

    case WRITE_10:

    case WRITE_VERIFY_10:

    case WRITE_12:

    case WRITE_VERIFY_12:

    case WRITE_16:

    case WRITE_VERIFY_16:

    case COPY:

    case COPY_VERIFY:

    case COMPARE:

    case CHANGE_DEFINITION:

    case LOG_SELECT:

    case MODE_SELECT:

    case MODE_SELECT_10:

    case SEND_DIAGNOSTIC:

    case WRITE_BUFFER:

    case FORMAT_UNIT:

    case REASSIGN_BLOCKS:

    case SEARCH_EQUAL:

    case SEARCH_HIGH:

    case SEARCH_LOW:

    case UPDATE_BLOCK:

    case WRITE_LONG_10:

    case WRITE_SAME_10:

    case WRITE_SAME_16:

    case UNMAP:

    case SEARCH_HIGH_12:

    case SEARCH_EQUAL_12:

    case SEARCH_LOW_12:

    case MEDIUM_SCAN:

    case SEND_VOLUME_TAG:

    case SEND_CUE_SHEET:

    case SEND_DVD_STRUCTURE:

    case PERSISTENT_RESERVE_OUT:

    case MAINTENANCE_OUT:


        cmd->mode = SCSI_XFER_TO_DEV;

        break;

    default:

        cmd->mode = SCSI_XFER_FROM_DEV;

        break;

    }

}