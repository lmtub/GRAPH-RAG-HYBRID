static int blkverify_open(BlockDriverState *bs, QDict *options, int flags,

                          Error **errp)

{

    BDRVBlkverifyState *s = bs->opaque;

    QemuOpts *opts;

    Error *local_err = NULL;

    int ret;



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;




    /* Open the raw file */

    bs->file = bdrv_open_child(qemu_opt_get(opts, "x-raw"), options, "raw",

                               bs, &child_file, false, &local_err);

    if (local_err) {

        ret = -EINVAL;

        error_propagate(errp, local_err);

        goto fail;




    /* Open the test file */

    s->test_file = bdrv_open_child(qemu_opt_get(opts, "x-image"), options,

                                   "test", bs, &child_format, false,

                                   &local_err);

    if (local_err) {

        ret = -EINVAL;

        error_propagate(errp, local_err);

        goto fail;




    ret = 0;

fail:




    qemu_opts_del(opts);

    return ret;
