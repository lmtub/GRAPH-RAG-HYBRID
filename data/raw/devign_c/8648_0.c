int bdrv_create_file(const char *filename, QemuOpts *opts, Error **errp)

{

    BlockDriver *drv;

    Error *local_err = NULL;

    int ret;



    drv = bdrv_find_protocol(filename, true, errp);

    if (drv == NULL) {

        return -ENOENT;

    }



    ret = bdrv_create(drv, filename, opts, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

    }

    return ret;

}
