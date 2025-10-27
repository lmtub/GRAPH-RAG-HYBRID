static bool cmd_write_multiple(IDEState *s, uint8_t cmd)

{

    bool lba48 = (cmd == WIN_MULTWRITE_EXT);

    int n;



    if (!s->bs || !s->mult_sectors) {

        ide_abort_command(s);

        return true;

    }



    ide_cmd_lba48_transform(s, lba48);



    s->req_nb_sectors = s->mult_sectors;

    n = MIN(s->nsector, s->req_nb_sectors);



    s->status = SEEK_STAT | READY_STAT;

    ide_transfer_start(s, s->io_buffer, 512 * n, ide_sector_write);



    s->media_changed = 1;



    return false;

}
