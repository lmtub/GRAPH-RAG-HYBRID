static QEMUFile *qemu_fopen_bdrv(BlockDriverState *bs, int64_t offset, int is_writable)

{

    QEMUFile *f;



    f = qemu_mallocz(sizeof(QEMUFile));

    if (!f)

        return NULL;

    f->is_file = 0;

    f->bs = bs;

    f->is_writable = is_writable;

    f->base_offset = offset;

    return f;

}
