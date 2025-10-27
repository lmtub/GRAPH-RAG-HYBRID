int sd_do_command(SDState *sd, SDRequest *req,

                  uint8_t *response) {

    uint32_t last_status = sd->card_status;

    sd_rsp_type_t rtype;

    int rsplen;



    if (!sd->bdrv || !bdrv_is_inserted(sd->bdrv) || !sd->enable) {

        return 0;

    }



    if (sd_req_crc_validate(req)) {

        sd->card_status &= ~COM_CRC_ERROR;

        return 0;

    }



    sd->card_status &= ~CARD_STATUS_B;

    sd_set_status(sd);



    if (last_status & CARD_IS_LOCKED) {

        if (!cmd_valid_while_locked(sd, req)) {

            sd->card_status |= ILLEGAL_COMMAND;

            fprintf(stderr, "SD: Card is locked\n");

            return 0;

        }

    }



    if (last_status & APP_CMD) {

        rtype = sd_app_command(sd, *req);

        sd->card_status &= ~APP_CMD;

    } else

        rtype = sd_normal_command(sd, *req);



    sd->current_cmd = req->cmd;



    switch (rtype) {

    case sd_r1:

    case sd_r1b:

        sd_response_r1_make(sd, response, last_status);

        rsplen = 4;

        break;



    case sd_r2_i:

        memcpy(response, sd->cid, sizeof(sd->cid));

        rsplen = 16;

        break;



    case sd_r2_s:

        memcpy(response, sd->csd, sizeof(sd->csd));

        rsplen = 16;

        break;



    case sd_r3:

        sd_response_r3_make(sd, response);

        rsplen = 4;

        break;



    case sd_r6:

        sd_response_r6_make(sd, response);

        rsplen = 4;

        break;



    case sd_r7:

        sd_response_r7_make(sd, response);

        rsplen = 4;

        break;



    case sd_r0:

    default:

        rsplen = 0;

        break;

    }



    if (sd->card_status & ILLEGAL_COMMAND)

        rsplen = 0;



#ifdef DEBUG_SD

    if (rsplen) {

        int i;

        DPRINTF("Response:");

        for (i = 0; i < rsplen; i++)

            printf(" %02x", response[i]);

        printf(" state %d\n", sd->state);

    } else {

        DPRINTF("No response %d\n", sd->state);

    }

#endif



    return rsplen;

}
