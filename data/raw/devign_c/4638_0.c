static void cdrom_change_cb(void *opaque)

{

    IDEState *s = opaque;

    uint64_t nb_sectors;



    /* XXX: send interrupt too */

    bdrv_get_geometry(s->bs, &nb_sectors);

    s->nb_sectors = nb_sectors;

}
