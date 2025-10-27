static int scsi_disk_emulate_command(SCSIDiskReq *r, uint8_t *outbuf)

{

    SCSIRequest *req = &r->req;

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, req->dev);

    uint64_t nb_sectors;

    int buflen = 0;

    int ret;



    switch (req->cmd.buf[0]) {

    case TEST_UNIT_READY:

        if (!bdrv_is_inserted(s->bs))

            goto not_ready;

	break;

    case REQUEST_SENSE:

        if (req->cmd.xfer < 4)

            goto illegal_request;

        memset(outbuf, 0, 4);

        buflen = 4;

        if (s->sense.key == NOT_READY && req->cmd.xfer >= 18) {

            memset(outbuf, 0, 18);

            buflen = 18;

            outbuf[7] = 10;

            /* asc 0x3a, ascq 0: Medium not present */

            outbuf[12] = 0x3a;

            outbuf[13] = 0;

        }

        outbuf[0] = 0xf0;

        outbuf[1] = 0;

        outbuf[2] = s->sense.key;

        scsi_disk_clear_sense(s);

        break;

    case INQUIRY:

        buflen = scsi_disk_emulate_inquiry(req, outbuf);

        if (buflen < 0)

            goto illegal_request;

	break;

    case MODE_SENSE:

    case MODE_SENSE_10:

        buflen = scsi_disk_emulate_mode_sense(req, outbuf);

        if (buflen < 0)

            goto illegal_request;

        break;

    case READ_TOC:

        buflen = scsi_disk_emulate_read_toc(req, outbuf);

        if (buflen < 0)

            goto illegal_request;

        break;

    case RESERVE:

        if (req->cmd.buf[1] & 1)

            goto illegal_request;

        break;

    case RESERVE_10:

        if (req->cmd.buf[1] & 3)

            goto illegal_request;

        break;

    case RELEASE:

        if (req->cmd.buf[1] & 1)

            goto illegal_request;

        break;

    case RELEASE_10:

        if (req->cmd.buf[1] & 3)

            goto illegal_request;

        break;

    case START_STOP:

        if (s->drive_kind == SCSI_CD && (req->cmd.buf[4] & 2)) {

            /* load/eject medium */

            bdrv_eject(s->bs, !(req->cmd.buf[4] & 1));

        }

	break;

    case ALLOW_MEDIUM_REMOVAL:

        bdrv_set_locked(s->bs, req->cmd.buf[4] & 1);

	break;

    case READ_CAPACITY:

        /* The normal LEN field for this command is zero.  */

	memset(outbuf, 0, 8);

	bdrv_get_geometry(s->bs, &nb_sectors);

        if (!nb_sectors)

            goto not_ready;

        nb_sectors /= s->cluster_size;

        /* Returned value is the address of the last sector.  */

        nb_sectors--;

        /* Remember the new size for read/write sanity checking. */

        s->max_lba = nb_sectors;

        /* Clip to 2TB, instead of returning capacity modulo 2TB. */

        if (nb_sectors > UINT32_MAX)

            nb_sectors = UINT32_MAX;

        outbuf[0] = (nb_sectors >> 24) & 0xff;

        outbuf[1] = (nb_sectors >> 16) & 0xff;

        outbuf[2] = (nb_sectors >> 8) & 0xff;

        outbuf[3] = nb_sectors & 0xff;

        outbuf[4] = 0;

        outbuf[5] = 0;

        outbuf[6] = s->cluster_size * 2;

        outbuf[7] = 0;

        buflen = 8;

	break;

    case SYNCHRONIZE_CACHE:

        ret = bdrv_flush(s->bs);

        if (ret < 0) {

            if (scsi_handle_rw_error(r, -ret, SCSI_REQ_STATUS_RETRY_FLUSH)) {

                return -1;

            }

        }

        break;

    case GET_CONFIGURATION:

        memset(outbuf, 0, 8);

        /* ??? This should probably return much more information.  For now

           just return the basic header indicating the CD-ROM profile.  */

        outbuf[7] = 8; // CD-ROM

        buflen = 8;

        break;

    case SERVICE_ACTION_IN:

        /* Service Action In subcommands. */

        if ((req->cmd.buf[1] & 31) == 0x10) {

            DPRINTF("SAI READ CAPACITY(16)\n");

            memset(outbuf, 0, req->cmd.xfer);

            bdrv_get_geometry(s->bs, &nb_sectors);

            if (!nb_sectors)

                goto not_ready;

            nb_sectors /= s->cluster_size;

            /* Returned value is the address of the last sector.  */

            nb_sectors--;

            /* Remember the new size for read/write sanity checking. */

            s->max_lba = nb_sectors;

            outbuf[0] = (nb_sectors >> 56) & 0xff;

            outbuf[1] = (nb_sectors >> 48) & 0xff;

            outbuf[2] = (nb_sectors >> 40) & 0xff;

            outbuf[3] = (nb_sectors >> 32) & 0xff;

            outbuf[4] = (nb_sectors >> 24) & 0xff;

            outbuf[5] = (nb_sectors >> 16) & 0xff;

            outbuf[6] = (nb_sectors >> 8) & 0xff;

            outbuf[7] = nb_sectors & 0xff;

            outbuf[8] = 0;

            outbuf[9] = 0;

            outbuf[10] = s->cluster_size * 2;

            outbuf[11] = 0;

            outbuf[12] = 0;

            outbuf[13] = get_physical_block_exp(&s->qdev.conf);



            /* set TPE bit if the format supports discard */

            if (s->qdev.conf.discard_granularity) {

                outbuf[14] = 0x80;

            }



            /* Protection, exponent and lowest lba field left blank. */

            buflen = req->cmd.xfer;

            break;

        }

        DPRINTF("Unsupported Service Action In\n");

        goto illegal_request;

    case REPORT_LUNS:

        if (req->cmd.xfer < 16)

            goto illegal_request;

        memset(outbuf, 0, 16);

        outbuf[3] = 8;

        buflen = 16;

        break;

    case VERIFY:

        break;

    case REZERO_UNIT:

        DPRINTF("Rezero Unit\n");

        if (!bdrv_is_inserted(s->bs)) {

            goto not_ready;

        }

        break;

    default:

        goto illegal_request;

    }

    scsi_req_set_status(r, GOOD, NO_SENSE);

    return buflen;



not_ready:

    scsi_command_complete(r, CHECK_CONDITION, NOT_READY);

    return -1;



illegal_request:

    scsi_command_complete(r, CHECK_CONDITION, ILLEGAL_REQUEST);

    return -1;

}
