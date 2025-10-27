static sd_rsp_type_t sd_app_command(SDState *sd,

                                    SDRequest req)

{

    DPRINTF("ACMD%d 0x%08x\n", req.cmd, req.arg);

    switch (req.cmd) {

    case 6:	/* ACMD6:  SET_BUS_WIDTH */

        switch (sd->state) {

        case sd_transfer_state:

            sd->sd_status[0] &= 0x3f;

            sd->sd_status[0] |= (req.arg & 0x03) << 6;

            return sd_r1;



        default:

            break;

        }

        break;



    case 13:	/* ACMD13: SD_STATUS */

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_sendingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 22:	/* ACMD22: SEND_NUM_WR_BLOCKS */

        switch (sd->state) {

        case sd_transfer_state:

            *(uint32_t *) sd->data = sd->blk_written;



            sd->state = sd_sendingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    case 23:	/* ACMD23: SET_WR_BLK_ERASE_COUNT */

        switch (sd->state) {

        case sd_transfer_state:

            return sd_r1;



        default:

            break;

        }

        break;



    case 41:	/* ACMD41: SD_APP_OP_COND */

        if (sd->spi) {

            /* SEND_OP_CMD */

            sd->state = sd_transfer_state;

            return sd_r1;

        }

        switch (sd->state) {

        case sd_idle_state:

            /* We accept any voltage.  10000 V is nothing.  */

            if (req.arg)

                sd->state = sd_ready_state;



            return sd_r3;



        default:

            break;

        }

        break;



    case 42:	/* ACMD42: SET_CLR_CARD_DETECT */

        switch (sd->state) {

        case sd_transfer_state:

            /* Bringing in the 50KOhm pull-up resistor... Done.  */

            return sd_r1;



        default:

            break;

        }

        break;



    case 51:	/* ACMD51: SEND_SCR */

        switch (sd->state) {

        case sd_transfer_state:

            sd->state = sd_sendingdata_state;

            sd->data_start = 0;

            sd->data_offset = 0;

            return sd_r1;



        default:

            break;

        }

        break;



    default:

        /* Fall back to standard commands.  */

        sd->card_status &= ~APP_CMD;

        return sd_normal_command(sd, req);

    }



    fprintf(stderr, "SD: ACMD%i in a wrong state\n", req.cmd);

    return sd_illegal;

}
