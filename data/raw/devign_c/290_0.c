void sd_write_data(SDState *sd, uint8_t value)

{

    int i;



    if (!sd->bdrv || !bdrv_is_inserted(sd->bdrv) || !sd->enable)

        return;



    if (sd->state != sd_receivingdata_state) {

        fprintf(stderr, "sd_write_data: not in Receiving-Data state\n");

        return;

    }



    if (sd->card_status & (ADDRESS_ERROR | WP_VIOLATION))

        return;



    switch (sd->current_cmd) {

    case 24:	/* CMD24:  WRITE_SINGLE_BLOCK */

        sd->data[sd->data_offset ++] = value;

        if (sd->data_offset >= sd->blk_len) {

            /* TODO: Check CRC before committing */

            sd->state = sd_programming_state;

            BLK_WRITE_BLOCK(sd->data_start, sd->data_offset);

            sd->blk_written ++;

            sd->csd[14] |= 0x40;

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

        }

        break;



    case 25:	/* CMD25:  WRITE_MULTIPLE_BLOCK */

        if (sd->data_offset == 0) {

            /* Start of the block - let's check the address is valid */

            if (sd->data_start + sd->blk_len > sd->size) {

                sd->card_status |= ADDRESS_ERROR;

                break;

            }

            if (sd_wp_addr(sd, sd->data_start)) {

                sd->card_status |= WP_VIOLATION;

                break;

            }

        }

        sd->data[sd->data_offset++] = value;

        if (sd->data_offset >= sd->blk_len) {

            /* TODO: Check CRC before committing */

            sd->state = sd_programming_state;

            BLK_WRITE_BLOCK(sd->data_start, sd->data_offset);

            sd->blk_written++;

            sd->data_start += sd->blk_len;

            sd->data_offset = 0;

            sd->csd[14] |= 0x40;



            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_receivingdata_state;

        }

        break;



    case 26:	/* CMD26:  PROGRAM_CID */

        sd->data[sd->data_offset ++] = value;

        if (sd->data_offset >= sizeof(sd->cid)) {

            /* TODO: Check CRC before committing */

            sd->state = sd_programming_state;

            for (i = 0; i < sizeof(sd->cid); i ++)

                if ((sd->cid[i] | 0x00) != sd->data[i])

                    sd->card_status |= CID_CSD_OVERWRITE;



            if (!(sd->card_status & CID_CSD_OVERWRITE))

                for (i = 0; i < sizeof(sd->cid); i ++) {

                    sd->cid[i] |= 0x00;

                    sd->cid[i] &= sd->data[i];

                }

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

        }

        break;



    case 27:	/* CMD27:  PROGRAM_CSD */

        sd->data[sd->data_offset ++] = value;

        if (sd->data_offset >= sizeof(sd->csd)) {

            /* TODO: Check CRC before committing */

            sd->state = sd_programming_state;

            for (i = 0; i < sizeof(sd->csd); i ++)

                if ((sd->csd[i] | sd_csd_rw_mask[i]) !=

                    (sd->data[i] | sd_csd_rw_mask[i]))

                    sd->card_status |= CID_CSD_OVERWRITE;



            /* Copy flag (OTP) & Permanent write protect */

            if (sd->csd[14] & ~sd->data[14] & 0x60)

                sd->card_status |= CID_CSD_OVERWRITE;



            if (!(sd->card_status & CID_CSD_OVERWRITE))

                for (i = 0; i < sizeof(sd->csd); i ++) {

                    sd->csd[i] |= sd_csd_rw_mask[i];

                    sd->csd[i] &= sd->data[i];

                }

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

        }

        break;



    case 42:	/* CMD42:  LOCK_UNLOCK */

        sd->data[sd->data_offset ++] = value;

        if (sd->data_offset >= sd->blk_len) {

            /* TODO: Check CRC before committing */

            sd->state = sd_programming_state;

            sd_lock_command(sd);

            /* Bzzzzzzztt .... Operation complete.  */

            sd->state = sd_transfer_state;

        }

        break;



    case 56:	/* CMD56:  GEN_CMD */

        sd->data[sd->data_offset ++] = value;

        if (sd->data_offset >= sd->blk_len) {

            APP_WRITE_BLOCK(sd->data_start, sd->data_offset);

            sd->state = sd_transfer_state;

        }

        break;



    default:

        fprintf(stderr, "sd_write_data: unknown command\n");

        break;

    }

}
