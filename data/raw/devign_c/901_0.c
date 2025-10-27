static void megasas_handle_frame(MegasasState *s, uint64_t frame_addr,

                                 uint32_t frame_count)

{

    uint8_t frame_status = MFI_STAT_INVALID_CMD;

    uint64_t frame_context;

    MegasasCmd *cmd;



    /*

     * Always read 64bit context, top bits will be

     * masked out if required in megasas_enqueue_frame()

     */

    frame_context = megasas_frame_get_context(s, frame_addr);



    cmd = megasas_enqueue_frame(s, frame_addr, frame_context, frame_count);

    if (!cmd) {

        /* reply queue full */

        trace_megasas_frame_busy(frame_addr);

        megasas_frame_set_scsi_status(s, frame_addr, BUSY);

        megasas_frame_set_cmd_status(s, frame_addr, MFI_STAT_SCSI_DONE_WITH_ERROR);

        megasas_complete_frame(s, frame_context);

        s->event_count++;

        return;

    }

    switch (cmd->frame->header.frame_cmd) {

    case MFI_CMD_INIT:

        frame_status = megasas_init_firmware(s, cmd);

        break;

    case MFI_CMD_DCMD:

        frame_status = megasas_handle_dcmd(s, cmd);

        break;

    case MFI_CMD_ABORT:

        frame_status = megasas_handle_abort(s, cmd);

        break;

    case MFI_CMD_PD_SCSI_IO:

        frame_status = megasas_handle_scsi(s, cmd, 0);

        break;

    case MFI_CMD_LD_SCSI_IO:

        frame_status = megasas_handle_scsi(s, cmd, 1);

        break;

    case MFI_CMD_LD_READ:

    case MFI_CMD_LD_WRITE:

        frame_status = megasas_handle_io(s, cmd);

        break;

    default:

        trace_megasas_unhandled_frame_cmd(cmd->index,

                                          cmd->frame->header.frame_cmd);

        s->event_count++;

        break;

    }

    if (frame_status != MFI_STAT_INVALID_STATUS) {

        if (cmd->frame) {

            cmd->frame->header.cmd_status = frame_status;

        } else {

            megasas_frame_set_cmd_status(s, frame_addr, frame_status);

        }

        megasas_unmap_frame(s, cmd);

        megasas_complete_frame(s, cmd->context);

    }

}
