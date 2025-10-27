static int qcow2_create(const char *filename, QemuOpts *opts, Error **errp)

{

    char *backing_file = NULL;

    char *backing_fmt = NULL;

    char *buf = NULL;

    uint64_t size = 0;

    int flags = 0;

    size_t cluster_size = DEFAULT_CLUSTER_SIZE;

    PreallocMode prealloc;

    int version;

    uint64_t refcount_bits;

    int refcount_order;

    const char *encryptfmt = NULL;

    Error *local_err = NULL;

    int ret;



    /* Read out options */

    size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                    BDRV_SECTOR_SIZE);

    backing_file = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FILE);

    backing_fmt = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FMT);

    encryptfmt = qemu_opt_get_del(opts, BLOCK_OPT_ENCRYPT_FORMAT);

    if (encryptfmt) {

        if (qemu_opt_get_del(opts, BLOCK_OPT_ENCRYPT)) {

            error_setg(errp, "Options " BLOCK_OPT_ENCRYPT " and "

                       BLOCK_OPT_ENCRYPT_FORMAT " are mutually exclusive");

            ret = -EINVAL;

            goto finish;

        }

    } else if (qemu_opt_get_bool_del(opts, BLOCK_OPT_ENCRYPT, false)) {

        encryptfmt = "aes";

    }

    cluster_size = qcow2_opt_get_cluster_size_del(opts, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto finish;

    }

    buf = qemu_opt_get_del(opts, BLOCK_OPT_PREALLOC);

    prealloc = qapi_enum_parse(PreallocMode_lookup, buf,

                               PREALLOC_MODE__MAX, PREALLOC_MODE_OFF,

                               &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto finish;

    }



    version = qcow2_opt_get_version_del(opts, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto finish;

    }



    if (qemu_opt_get_bool_del(opts, BLOCK_OPT_LAZY_REFCOUNTS, false)) {

        flags |= BLOCK_FLAG_LAZY_REFCOUNTS;

    }



    if (backing_file && prealloc != PREALLOC_MODE_OFF) {

        error_setg(errp, "Backing file and preallocation cannot be used at "

                   "the same time");

        ret = -EINVAL;

        goto finish;

    }



    if (version < 3 && (flags & BLOCK_FLAG_LAZY_REFCOUNTS)) {

        error_setg(errp, "Lazy refcounts only supported with compatibility "

                   "level 1.1 and above (use compat=1.1 or greater)");

        ret = -EINVAL;

        goto finish;

    }



    refcount_bits = qcow2_opt_get_refcount_bits_del(opts, version, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto finish;

    }



    refcount_order = ctz32(refcount_bits);



    ret = qcow2_create2(filename, size, backing_file, backing_fmt, flags,

                        cluster_size, prealloc, opts, version, refcount_order,

                        encryptfmt, &local_err);

    error_propagate(errp, local_err);



finish:

    g_free(backing_file);

    g_free(backing_fmt);

    g_free(buf);

    return ret;

}
