void qmp_blockdev_add(BlockdevOptions *options, Error **errp)

{

    QmpOutputVisitor *ov = qmp_output_visitor_new();

    QObject *obj;

    QDict *qdict;

    Error *local_err = NULL;



    /* Require an ID in the top level */

    if (!options->has_id) {

        error_setg(errp, "Block device needs an ID");

        goto fail;

    }



    /* TODO Sort it out in raw-posix and drive_init: Reject aio=native with

     * cache.direct=false instead of silently switching to aio=threads, except

     * if called from drive_init.

     *

     * For now, simply forbidding the combination for all drivers will do. */

    if (options->has_aio && options->aio == BLOCKDEV_AIO_OPTIONS_NATIVE) {

        bool direct = options->cache->has_direct && options->cache->direct;

        if (!options->has_cache && !direct) {

            error_setg(errp, "aio=native requires cache.direct=true");

            goto fail;

        }

    }



    visit_type_BlockdevOptions(qmp_output_get_visitor(ov),

                               &options, NULL, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto fail;

    }



    obj = qmp_output_get_qobject(ov);

    qdict = qobject_to_qdict(obj);



    qdict_flatten(qdict);



    blockdev_init(NULL, qdict, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto fail;

    }



fail:

    qmp_output_visitor_cleanup(ov);

}
