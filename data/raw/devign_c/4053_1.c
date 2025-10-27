static void cmd_read_cd(IDEState *s, uint8_t* buf)

{

    int nb_sectors, lba, transfer_request;



    nb_sectors = (buf[6] << 16) | (buf[7] << 8) | buf[8];

    lba = ube32_to_cpu(buf + 2);



    if (nb_sectors == 0) {

        ide_atapi_cmd_ok(s);

        return;

    }



    transfer_request = buf[9];

    switch(transfer_request & 0xf8) {

    case 0x00:

        /* nothing */

        ide_atapi_cmd_ok(s);

        break;

    case 0x10:

        /* normal read */

        ide_atapi_cmd_read(s, lba, nb_sectors, 2048);

        break;

    case 0xf8:

        /* read all data */

        ide_atapi_cmd_read(s, lba, nb_sectors, 2352);

        break;

    default:

        ide_atapi_cmd_error(s, ILLEGAL_REQUEST,

                            ASC_INV_FIELD_IN_CMD_PACKET);

        break;

    }

}
