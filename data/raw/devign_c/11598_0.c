static bool cmd_identify(IDEState *s, uint8_t cmd)

{

    if (s->bs && s->drive_kind != IDE_CD) {

        if (s->drive_kind != IDE_CFATA) {

            ide_identify(s);

        } else {

            ide_cfata_identify(s);

        }

        s->status = READY_STAT | SEEK_STAT;

        ide_transfer_start(s, s->io_buffer, 512, ide_transfer_stop);

        ide_set_irq(s->bus);

        return false;

    } else {

        if (s->drive_kind == IDE_CD) {

            ide_set_signature(s);

        }

        ide_abort_command(s);

    }



    return true;

}
