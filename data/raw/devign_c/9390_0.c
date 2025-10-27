static inline bool media_is_cd(SCSIDiskState *s)

{

    uint64_t nb_sectors;

    if (s->qdev.type != TYPE_ROM) {

        return false;

    }

    if (!bdrv_is_inserted(s->qdev.conf.bs)) {

        return false;

    }

    bdrv_get_geometry(s->qdev.conf.bs, &nb_sectors);

    return nb_sectors <= CD_MAX_SECTORS;

}
