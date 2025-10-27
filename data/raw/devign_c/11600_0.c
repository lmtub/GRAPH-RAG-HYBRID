static void cmd_read_cdvd_capacity(IDEState *s, uint8_t* buf)

{

    uint64_t total_sectors = s->nb_sectors >> 2;



    if (total_sectors == 0) {

        ide_atapi_cmd_error(s, SENSE_NOT_READY, ASC_MEDIUM_NOT_PRESENT);

        return;

    }



    /* NOTE: it is really the number of sectors minus 1 */

    cpu_to_ube32(buf, total_sectors - 1);

    cpu_to_ube32(buf + 4, 2048);

    ide_atapi_cmd_reply(s, 8, 8);

}
