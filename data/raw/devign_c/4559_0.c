int64_t qemu_file_set_rate_limit(QEMUFile *f, int64_t new_rate)

{

    /* any failed or completed migration keeps its state to allow probing of

     * migration data, but has no associated file anymore */

    if (f && f->ops->set_rate_limit)

        return f->ops->set_rate_limit(f->opaque, new_rate);



    return 0;

}
