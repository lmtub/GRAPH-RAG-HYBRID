int bdrv_create(BlockDriver *drv, const char* filename,

                QemuOpts *opts, Error **errp)

{

    int ret;



    Coroutine *co;

    CreateCo cco = {

        .drv = drv,

        .filename = g_strdup(filename),

        .opts = opts,

        .ret = NOT_DONE,

        .err = NULL,

    };



    if (!drv->bdrv_create) {

        error_setg(errp, "Driver '%s' does not support image creation", drv->format_name);

        ret = -ENOTSUP;

        goto out;

    }



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_create_co_entry(&cco);

    } else {

        co = qemu_coroutine_create(bdrv_create_co_entry);

        qemu_coroutine_enter(co, &cco);

        while (cco.ret == NOT_DONE) {

            aio_poll(qemu_get_aio_context(), true);

        }

    }



    ret = cco.ret;

    if (ret < 0) {

        if (cco.err) {

            error_propagate(errp, cco.err);

        } else {

            error_setg_errno(errp, -ret, "Could not create image");

        }

    }



out:

    g_free(cco.filename);

    return ret;

}
