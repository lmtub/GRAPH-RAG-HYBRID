static int scsi_read_dvd_structure(SCSIDiskState *s, SCSIDiskReq *r,

                                   uint8_t *outbuf)

{

    static const int rds_caps_size[5] = {

        [0] = 2048 + 4,

        [1] = 4 + 4,

        [3] = 188 + 4,

        [4] = 2048 + 4,

    };



    uint8_t media = r->req.cmd.buf[1];

    uint8_t layer = r->req.cmd.buf[6];

    uint8_t format = r->req.cmd.buf[7];

    int size = -1;



    if (s->qdev.type != TYPE_ROM) {

        return -1;

    }

    if (media != 0) {

        scsi_check_condition(r, SENSE_CODE(INVALID_FIELD));

        return -1;

    }



    if (format != 0xff) {

        if (s->tray_open || !bdrv_is_inserted(s->qdev.conf.bs)) {

            scsi_check_condition(r, SENSE_CODE(NO_MEDIUM));

            return -1;

        }

        if (media_is_cd(s)) {

            scsi_check_condition(r, SENSE_CODE(INCOMPATIBLE_FORMAT));

            return -1;

        }

        if (format >= ARRAY_SIZE(rds_caps_size)) {

            return -1;

        }

        size = rds_caps_size[format];

        memset(outbuf, 0, size);

    }



    switch (format) {

    case 0x00: {

        /* Physical format information */

        uint64_t nb_sectors;

        if (layer != 0) {

            goto fail;

        }

        bdrv_get_geometry(s->qdev.conf.bs, &nb_sectors);



        outbuf[4] = 1;   /* DVD-ROM, part version 1 */

        outbuf[5] = 0xf; /* 120mm disc, minimum rate unspecified */

        outbuf[6] = 1;   /* one layer, read-only (per MMC-2 spec) */

        outbuf[7] = 0;   /* default densities */



        stl_be_p(&outbuf[12], (nb_sectors >> 2) - 1); /* end sector */

        stl_be_p(&outbuf[16], (nb_sectors >> 2) - 1); /* l0 end sector */

        break;

    }



    case 0x01: /* DVD copyright information, all zeros */

        break;



    case 0x03: /* BCA information - invalid field for no BCA info */

        return -1;



    case 0x04: /* DVD disc manufacturing information, all zeros */

        break;



    case 0xff: { /* List capabilities */

        int i;

        size = 4;

        for (i = 0; i < ARRAY_SIZE(rds_caps_size); i++) {

            if (!rds_caps_size[i]) {

                continue;

            }

            outbuf[size] = i;

            outbuf[size + 1] = 0x40; /* Not writable, readable */

            stw_be_p(&outbuf[size + 2], rds_caps_size[i]);

            size += 4;

        }

        break;

     }



    default:

        return -1;

    }



    /* Size of buffer, not including 2 byte size field */

    stw_be_p(outbuf, size - 2);

    return size;



fail:

    return -1;

}
