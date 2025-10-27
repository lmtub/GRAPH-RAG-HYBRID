int ide_get_geometry(BusState *bus, int unit,

                     int16_t *cyls, int8_t *heads, int8_t *secs)

{

    IDEState *s = &DO_UPCAST(IDEBus, qbus, bus)->ifs[unit];



    if (s->drive_kind != IDE_HD || !s->bs) {

        return -1;

    }



    *cyls = s->cylinders;

    *heads = s->heads;

    *secs = s->sectors;

    return 0;

}
