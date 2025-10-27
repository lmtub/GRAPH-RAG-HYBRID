static bool cmd_write_pio(IDEState *s, uint8_t cmd)

{

    bool lba48 = (cmd == WIN_WRITE_EXT);



    if (!s->bs) {

        ide_abort_command(s);

        return true;

    }



    ide_cmd_lba48_transform(s, lba48);



    s->req_nb_sectors = 1;

    s->status = SEEK_STAT | READY_STAT;

    ide_transfer_start(s, s->io_buffer, 512, ide_sector_write);



    s->media_changed = 1;



    return false;

}
