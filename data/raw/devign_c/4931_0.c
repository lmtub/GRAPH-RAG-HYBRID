static int handle_cmd(AHCIState *s, int port, int slot)

{

    IDEState *ide_state;

    uint32_t opts;

    uint64_t tbl_addr;

    AHCICmdHdr *cmd;

    uint8_t *cmd_fis;

    dma_addr_t cmd_len;



    if (s->dev[port].port.ifs[0].status & (BUSY_STAT|DRQ_STAT)) {

        /* Engine currently busy, try again later */

        DPRINTF(port, "engine busy\n");

        return -1;

    }



    cmd = &((AHCICmdHdr *)s->dev[port].lst)[slot];



    if (!s->dev[port].lst) {

        DPRINTF(port, "error: lst not given but cmd handled");

        return -1;

    }



    /* remember current slot handle for later */

    s->dev[port].cur_cmd = cmd;



    opts = le32_to_cpu(cmd->opts);

    tbl_addr = le64_to_cpu(cmd->tbl_addr);



    cmd_len = 0x80;

    cmd_fis = dma_memory_map(s->as, tbl_addr, &cmd_len,

                             DMA_DIRECTION_FROM_DEVICE);



    if (!cmd_fis) {

        DPRINTF(port, "error: guest passed us an invalid cmd fis\n");

        return -1;

    }



    /* The device we are working for */

    ide_state = &s->dev[port].port.ifs[0];



    if (!ide_state->bs) {

        DPRINTF(port, "error: guest accessed unused port");

        goto out;

    }



    debug_print_fis(cmd_fis, 0x90);

    //debug_print_fis(cmd_fis, (opts & AHCI_CMD_HDR_CMD_FIS_LEN) * 4);



    switch (cmd_fis[0]) {

        case SATA_FIS_TYPE_REGISTER_H2D:

            break;

        default:

            DPRINTF(port, "unknown command cmd_fis[0]=%02x cmd_fis[1]=%02x "

                          "cmd_fis[2]=%02x\n", cmd_fis[0], cmd_fis[1],

                          cmd_fis[2]);

            goto out;

            break;

    }



    switch (cmd_fis[1]) {

        case SATA_FIS_REG_H2D_UPDATE_COMMAND_REGISTER:

            break;

        case 0:

            break;

        default:

            DPRINTF(port, "unknown command cmd_fis[0]=%02x cmd_fis[1]=%02x "

                          "cmd_fis[2]=%02x\n", cmd_fis[0], cmd_fis[1],

                          cmd_fis[2]);

            goto out;

            break;

    }



    switch (s->dev[port].port_state) {

        case STATE_RUN:

            if (cmd_fis[15] & ATA_SRST) {

                s->dev[port].port_state = STATE_RESET;

            }

            break;

        case STATE_RESET:

            if (!(cmd_fis[15] & ATA_SRST)) {

                ahci_reset_port(s, port);

            }

            break;

    }



    if (cmd_fis[1] == SATA_FIS_REG_H2D_UPDATE_COMMAND_REGISTER) {



        /* Check for NCQ command */

        if ((cmd_fis[2] == READ_FPDMA_QUEUED) ||

            (cmd_fis[2] == WRITE_FPDMA_QUEUED)) {

            process_ncq_command(s, port, cmd_fis, slot);

            goto out;

        }



        /* Decompose the FIS  */

        ide_state->nsector = (int64_t)((cmd_fis[13] << 8) | cmd_fis[12]);

        ide_state->feature = cmd_fis[3];

        if (!ide_state->nsector) {

            ide_state->nsector = 256;

        }



        if (ide_state->drive_kind != IDE_CD) {

            /*

             * We set the sector depending on the sector defined in the FIS.

             * Unfortunately, the spec isn't exactly obvious on this one.

             *

             * Apparently LBA48 commands set fis bytes 10,9,8,6,5,4 to the

             * 48 bit sector number. ATA_CMD_READ_DMA_EXT is an example for

             * such a command.

             *

             * Non-LBA48 commands however use 7[lower 4 bits],6,5,4 to define a

             * 28-bit sector number. ATA_CMD_READ_DMA is an example for such

             * a command.

             *

             * Since the spec doesn't explicitly state what each field should

             * do, I simply assume non-used fields as reserved and OR everything

             * together, independent of the command.

             */

            ide_set_sector(ide_state, ((uint64_t)cmd_fis[10] << 40)

                                    | ((uint64_t)cmd_fis[9] << 32)

                                    /* This is used for LBA48 commands */

                                    | ((uint64_t)cmd_fis[8] << 24)

                                    /* This is used for non-LBA48 commands */

                                    | ((uint64_t)(cmd_fis[7] & 0xf) << 24)

                                    | ((uint64_t)cmd_fis[6] << 16)

                                    | ((uint64_t)cmd_fis[5] << 8)

                                    | cmd_fis[4]);

        }



        /* Copy the ACMD field (ATAPI packet, if any) from the AHCI command

         * table to ide_state->io_buffer

         */

        if (opts & AHCI_CMD_ATAPI) {

            memcpy(ide_state->io_buffer, &cmd_fis[AHCI_COMMAND_TABLE_ACMD], 0x10);

            ide_state->lcyl = 0x14;

            ide_state->hcyl = 0xeb;

            debug_print_fis(ide_state->io_buffer, 0x10);

            ide_state->feature = IDE_FEATURE_DMA;

            s->dev[port].done_atapi_packet = false;

            /* XXX send PIO setup FIS */

        }



        ide_state->error = 0;



        /* Reset transferred byte counter */

        cmd->status = 0;



        /* We're ready to process the command in FIS byte 2. */

        ide_exec_cmd(&s->dev[port].port, cmd_fis[2]);

    }



out:

    dma_memory_unmap(s->as, cmd_fis, cmd_len, DMA_DIRECTION_FROM_DEVICE,

                     cmd_len);



    if (s->dev[port].port.ifs[0].status & (BUSY_STAT|DRQ_STAT)) {

        /* async command, complete later */

        s->dev[port].busy_slot = slot;

        return -1;

    }



    /* done handling the command */

    return 0;

}
