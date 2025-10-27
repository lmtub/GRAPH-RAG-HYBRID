uint8_t sd_read_data(SDState *sd)

{

    /* TODO: Append CRCs */

    uint8_t ret;

    int io_len;



    if (!sd->blk || !blk_is_inserted(sd->blk) || !sd->enable)

        return 0x00;



    if (sd->state != sd_sendingdata_state) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "sd_read_data: not in Sending-Data state\n");

        return 0x00;

    }



    if (sd->card_status & (ADDRESS_ERROR | WP_VIOLATION))

        return 0x00;



    io_len = (sd->ocr & (1 << 30)) ? 512 : sd->blk_len;



    switch (sd->current_cmd) {

    case 6:	/* CMD6:   SWITCH_FUNCTION */

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= 64)

            sd->state = sd_transfer_state;

        break;



    case 9:	/* CMD9:   SEND_CSD */

    case 10:	/* CMD10:  SEND_CID */

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= 16)

            sd->state = sd_transfer_state;

        break;



    case 11:	/* CMD11:  READ_DAT_UNTIL_STOP */

        if (sd->data_offset == 0)

            BLK_READ_BLOCK(sd->data_start, io_len);

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= io_len) {

            sd->data_start += io_len;

            sd->data_offset = 0;

            if (sd->data_start + io_len > sd->size) {

                sd->card_status |= ADDRESS_ERROR;

                break;

            }

        }

        break;



    case 13:	/* ACMD13: SD_STATUS */

        ret = sd->sd_status[sd->data_offset ++];



        if (sd->data_offset >= sizeof(sd->sd_status))

            sd->state = sd_transfer_state;

        break;



    case 17:	/* CMD17:  READ_SINGLE_BLOCK */

        if (sd->data_offset == 0)

            BLK_READ_BLOCK(sd->data_start, io_len);

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= io_len)

            sd->state = sd_transfer_state;

        break;



    case 18:	/* CMD18:  READ_MULTIPLE_BLOCK */

        if (sd->data_offset == 0)

            BLK_READ_BLOCK(sd->data_start, io_len);

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= io_len) {

            sd->data_start += io_len;

            sd->data_offset = 0;



            if (sd->multi_blk_cnt != 0) {

                if (--sd->multi_blk_cnt == 0) {

                    /* Stop! */

                    sd->state = sd_transfer_state;

                    break;

                }

            }



            if (sd->data_start + io_len > sd->size) {

                sd->card_status |= ADDRESS_ERROR;

                break;

            }

        }

        break;



    case 22:	/* ACMD22: SEND_NUM_WR_BLOCKS */

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= 4)

            sd->state = sd_transfer_state;

        break;



    case 30:	/* CMD30:  SEND_WRITE_PROT */

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= 4)

            sd->state = sd_transfer_state;

        break;



    case 51:	/* ACMD51: SEND_SCR */

        ret = sd->scr[sd->data_offset ++];



        if (sd->data_offset >= sizeof(sd->scr))

            sd->state = sd_transfer_state;

        break;



    case 56:	/* CMD56:  GEN_CMD */

        if (sd->data_offset == 0)

            APP_READ_BLOCK(sd->data_start, sd->blk_len);

        ret = sd->data[sd->data_offset ++];



        if (sd->data_offset >= sd->blk_len)

            sd->state = sd_transfer_state;

        break;



    default:

        qemu_log_mask(LOG_GUEST_ERROR, "sd_read_data: unknown command\n");

        return 0x00;

    }



    return ret;

}
