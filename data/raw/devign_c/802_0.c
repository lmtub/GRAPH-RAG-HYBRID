static sd_rsp_type_t sd_normal_command(SDState *sd,

                                       SDRequest req)

{

    uint32_t rca = 0x0000;

    uint64_t addr = (sd->ocr & (1 << 30)) ? (uint64_t) req.arg << 9 : req.arg;



    /* Not interpreting this as an app command */

    sd->card_status &= ~APP_CMD;



    if (sd_cmd_type[req.cmd] == sd_ac || sd_cmd_type[req.cmd] == sd_adtc)

        rca = req.arg >> 16;



    DPRINTF("CMD%d 0x%08x state %d\n", req.cmd, req.arg, sd->state);

    switch (req.cmd) {

    /* Basic commands (Class 0 and Class 1) */

    case 0:	/* CMD0:   GO_IDLE_STATE */

        switch (sd->state) {

        case sd_inactive_state:

            return sd->spi ? sd_r1 : sd_r0;



        default:

            sd->state = sd_idle_state;

            sd_reset(sd, sd->bdrv);

            return sd->spi ? sd_r1 : sd_r0;

        }

        break;



    case 1:	/* CMD1:   SEND_OP_CMD */

        if (!sd->spi)

            goto bad_cmd;



        sd->state = sd_transfer_state;

        return sd_r1;



    case 2:	/* CMD2:   ALL_SEND_CID */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->state) {

        case sd_ready_state:

            sd->state = sd_identification_state;

            return sd_r2_i;



        default:

            break;

        }

        break;



    case 3:	/* CMD3:   SEND_RELATIVE_ADDR */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->state) {

        case sd_identification_state:

        case sd_standby_state:

            sd->state = sd_standby_state;

            sd_set_rca(sd);

            return sd_r6;



        default:

            break;

        }

        break;



    case 4:	/* CMD4:   SEND_DSR */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->state) {

        case sd_standby_state:

            break;



        default:

            break;

        }

        break;



    case 5: /* CMD5: reserved for SDIO cards */

        return sd_illegal;



    case 6:	/* CMD6:   SWITCH_FUNCTION */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->mode) {

        case sd_data_transfer_mode:

            sd_function_switch(sd, req.arg);

            sd->state = sd_sendingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 7:	/* CMD7:   SELECT/DESELECT_CARD */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->state) {

        case sd_standby_state:

            if (sd->rca != rca)

                return sd_r0;



            sd->state = sd_transfer_state;

            return sd_r1b;



        case sd_transfer_state:

        case sd_sendingdata_state:

            if (sd->rca == rca)

                break;



            sd->state = sd_standby_state;

            return sd_r1b;



        case sd_disconnect_state:

            if (sd->rca != rca)

                return sd_r0;



            sd->state = sd_programming_state;

            return sd_r1b;



        case sd_programming_state:

            if (sd->rca == rca)

                break;



            sd->state = sd_disconnect_state;

            return sd_r1b;



        default:

            break;

        }

        break;



    case 8:	/* CMD8:   SEND_IF_COND */

        /* Physical Layer Specification Version 2.00 command */

        switch (sd->state) {

        case sd_idle_state:

            sd->vhs = 0;



            /* No response if not exactly one VHS bit is set.  */

            if (!(req.arg >> 8) || (req.arg >> ffs(req.arg & ~0xff)))

                return sd->spi ? sd_r7 : sd_r0;



            /* Accept.  */

            sd->vhs = req.arg;

            return sd_r7;



        default:

            break;

        }

        break;



    case 9:	/* CMD9:   SEND_CSD */

        switch (sd->state) {

        case sd_standby_state:

            if (sd->rca != rca)

                return sd_r0;



            return sd_r2_s;



        case sd_transfer_state:

            if (!sd->spi)

                break;

            sd->state = sd_sendingdata_state;

            memcpy(sd->data, sd->csd, 16);

            sd->data_start = addr;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 10:	/* CMD10:  SEND_CID */

        switch (sd->state) {

        case sd_standby_state:

            if (sd->rca != rca)

                return sd_r0;



            return sd_r2_i;



        case sd_transfer_state:

            if (!sd->spi)

                break;

            sd->state = sd_sendingdata_state;

            memcpy(sd->data, sd->cid, 16);

            sd->data_start = addr;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 11:	/* CMD11:  READ_DAT_UNTIL_STOP */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_sendingdata_state;

            sd->data_start = req.arg;

            sd->data_offset = 0;



            if (sd->data_start + sd->blk_len > sd->size)

                sd->card_status |= ADDRESS_ERROR;

            return sd_r0;



        default:

            break;

        }

        break;



    case 12:	/* CMD12:  STOP_TRANSMISSION */

        switch (sd->state) {

        case sd_sendingdata_state:

            sd->state = sd_transfer_state;

            return sd_r1b;



        case sd_receivingdata_state:

            sd->state = sd_programming_state;

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

            return sd_r1b;



        default:

            break;

        }

        break;



    case 13:	/* CMD13:  SEND_STATUS */

        switch (sd->mode) {

        case sd_data_transfer_mode:

            if (sd->rca != rca)

                return sd_r0;



            return sd_r1;



        default:

            break;

        }

        break;



    case 15:	/* CMD15:  GO_INACTIVE_STATE */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->mode) {

        case sd_data_transfer_mode:

            if (sd->rca != rca)

                return sd_r0;



            sd->state = sd_inactive_state;

            return sd_r0;



        default:

            break;

        }

        break;



    /* Block read commands (Classs 2) */

    case 16:	/* CMD16:  SET_BLOCKLEN */

        switch (sd->state) {

        case sd_transfer_state:

            if (req.arg > (1 << HWBLOCK_SHIFT))

                sd->card_status |= BLOCK_LEN_ERROR;

            else

                sd->blk_len = req.arg;



            return sd_r1;



        default:

            break;

        }

        break;



    case 17:	/* CMD17:  READ_SINGLE_BLOCK */

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_sendingdata_state;

            sd->data_start = addr;

            sd->data_offset = 0;



            if (sd->data_start + sd->blk_len > sd->size)

                sd->card_status |= ADDRESS_ERROR;

            return sd_r1;



        default:

            break;

        }

        break;



    case 18:	/* CMD18:  READ_MULTIPLE_BLOCK */

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_sendingdata_state;

            sd->data_start = addr;

            sd->data_offset = 0;



            if (sd->data_start + sd->blk_len > sd->size)

                sd->card_status |= ADDRESS_ERROR;

            return sd_r1;



        default:

            break;

        }

        break;



    /* Block write commands (Class 4) */

    case 24:	/* CMD24:  WRITE_SINGLE_BLOCK */

        if (sd->spi)

            goto unimplemented_cmd;

        switch (sd->state) {

        case sd_transfer_state:

            /* Writing in SPI mode not implemented.  */

            if (sd->spi)

                break;

            sd->state = sd_receivingdata_state;

            sd->data_start = addr;

            sd->data_offset = 0;

            sd->blk_written = 0;



            if (sd->data_start + sd->blk_len > sd->size)

                sd->card_status |= ADDRESS_ERROR;

            if (sd_wp_addr(sd, sd->data_start))

                sd->card_status |= WP_VIOLATION;

            if (sd->csd[14] & 0x30)

                sd->card_status |= WP_VIOLATION;

            return sd_r1;



        default:

            break;

        }

        break;



    case 25:	/* CMD25:  WRITE_MULTIPLE_BLOCK */

        if (sd->spi)

            goto unimplemented_cmd;

        switch (sd->state) {

        case sd_transfer_state:

            /* Writing in SPI mode not implemented.  */

            if (sd->spi)

                break;

            sd->state = sd_receivingdata_state;

            sd->data_start = addr;

            sd->data_offset = 0;

            sd->blk_written = 0;



            if (sd->data_start + sd->blk_len > sd->size)

                sd->card_status |= ADDRESS_ERROR;

            if (sd_wp_addr(sd, sd->data_start))

                sd->card_status |= WP_VIOLATION;

            if (sd->csd[14] & 0x30)

                sd->card_status |= WP_VIOLATION;

            return sd_r1;



        default:

            break;

        }

        break;



    case 26:	/* CMD26:  PROGRAM_CID */

        if (sd->spi)

            goto bad_cmd;

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_receivingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 27:	/* CMD27:  PROGRAM_CSD */

        if (sd->spi)

            goto unimplemented_cmd;

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_receivingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    /* Write protection (Class 6) */

    case 28:	/* CMD28:  SET_WRITE_PROT */

        switch (sd->state) {

        case sd_transfer_state:

            if (addr >= sd->size) {

                sd->card_status |= ADDRESS_ERROR;

                return sd_r1b;

            }



            sd->state = sd_programming_state;

            set_bit(sd_addr_to_wpnum(addr), sd->wp_groups);

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

            return sd_r1b;



        default:

            break;

        }

        break;



    case 29:	/* CMD29:  CLR_WRITE_PROT */

        switch (sd->state) {

        case sd_transfer_state:

            if (addr >= sd->size) {

                sd->card_status |= ADDRESS_ERROR;

                return sd_r1b;

            }



            sd->state = sd_programming_state;

            clear_bit(sd_addr_to_wpnum(addr), sd->wp_groups);

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

            return sd_r1b;



        default:

            break;

        }

        break;



    case 30:	/* CMD30:  SEND_WRITE_PROT */

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_sendingdata_state;

            *(uint32_t *) sd->data = sd_wpbits(sd, req.arg);

            sd->data_start = addr;

            sd->data_offset = 0;

            return sd_r1b;



        default:

            break;

        }

        break;



    /* Erase commands (Class 5) */

    case 32:	/* CMD32:  ERASE_WR_BLK_START */

        switch (sd->state) {

        case sd_transfer_state:

            sd->erase_start = req.arg;

            return sd_r1;



        default:

            break;

        }

        break;



    case 33:	/* CMD33:  ERASE_WR_BLK_END */

        switch (sd->state) {

        case sd_transfer_state:

            sd->erase_end = req.arg;

            return sd_r1;



        default:

            break;

        }

        break;



    case 38:	/* CMD38:  ERASE */

        switch (sd->state) {

        case sd_transfer_state:

            if (sd->csd[14] & 0x30) {

                sd->card_status |= WP_VIOLATION;

                return sd_r1b;

            }



            sd->state = sd_programming_state;

            sd_erase(sd);

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

            return sd_r1b;



        default:

            break;

        }

        break;



    /* Lock card commands (Class 7) */

    case 42:	/* CMD42:  LOCK_UNLOCK */

        if (sd->spi)

            goto unimplemented_cmd;

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_receivingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 52:

    case 53:

        /* CMD52, CMD53: reserved for SDIO cards

         * (see the SDIO Simplified Specification V2.0)

         * Handle as illegal command but do not complain

         * on stderr, as some OSes may use these in their

         * probing for presence of an SDIO card.

         */

        return sd_illegal;



    /* Application specific commands (Class 8) */

    case 55:	/* CMD55:  APP_CMD */

        if (sd->rca != rca)

            return sd_r0;



        sd->expecting_acmd = true;

        sd->card_status |= APP_CMD;

        return sd_r1;



    case 56:	/* CMD56:  GEN_CMD */

        fprintf(stderr, "SD: GEN_CMD 0x%08x\n", req.arg);



        switch (sd->state) {

        case sd_transfer_state:

            sd->data_offset = 0;

            if (req.arg & 1)

                sd->state = sd_sendingdata_state;

            else

                sd->state = sd_receivingdata_state;

            return sd_r1;



        default:

            break;

        }

        break;



    default:

    bad_cmd:

        fprintf(stderr, "SD: Unknown CMD%i\n", req.cmd);

        return sd_illegal;



    unimplemented_cmd:

        /* Commands that are recognised but not yet implemented in SPI mode.  */

        fprintf(stderr, "SD: CMD%i not implemented in SPI mode\n", req.cmd);

        return sd_illegal;

    }



    fprintf(stderr, "SD: CMD%i in a wrong state\n", req.cmd);

    return sd_illegal;

}
