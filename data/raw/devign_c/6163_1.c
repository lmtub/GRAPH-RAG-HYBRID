static int qcow2_update_options(BlockDriverState *bs, QDict *options,

                                int flags, Error **errp)

{

    BDRVQcow2State *s = bs->opaque;

    QemuOpts *opts = NULL;

    const char *opt_overlap_check, *opt_overlap_check_template;

    int overlap_check_template = 0;

    uint64_t l2_cache_size, refcount_cache_size;

    Qcow2Cache *l2_table_cache;

    Qcow2Cache *refcount_block_cache;

    uint64_t cache_clean_interval;

    bool use_lazy_refcounts;

    int i;

    Error *local_err = NULL;

    int ret;



    opts = qemu_opts_create(&qcow2_runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    /* get L2 table/refcount block cache size from command line options */

    read_cache_sizes(bs, opts, &l2_cache_size, &refcount_cache_size,

                     &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    l2_cache_size /= s->cluster_size;

    if (l2_cache_size < MIN_L2_CACHE_SIZE) {

        l2_cache_size = MIN_L2_CACHE_SIZE;

    }

    if (l2_cache_size > INT_MAX) {

        error_setg(errp, "L2 cache size too big");

        ret = -EINVAL;

        goto fail;

    }



    refcount_cache_size /= s->cluster_size;

    if (refcount_cache_size < MIN_REFCOUNT_CACHE_SIZE) {

        refcount_cache_size = MIN_REFCOUNT_CACHE_SIZE;

    }

    if (refcount_cache_size > INT_MAX) {

        error_setg(errp, "Refcount cache size too big");

        ret = -EINVAL;

        goto fail;

    }



    /* alloc L2 table/refcount block cache */

    l2_table_cache = qcow2_cache_create(bs, l2_cache_size);

    refcount_block_cache = qcow2_cache_create(bs, refcount_cache_size);

    if (l2_table_cache == NULL || refcount_block_cache == NULL) {

        error_setg(errp, "Could not allocate metadata caches");

        ret = -ENOMEM;

        goto fail;

    }



    /* New interval for cache cleanup timer */

    cache_clean_interval =

        qemu_opt_get_number(opts, QCOW2_OPT_CACHE_CLEAN_INTERVAL, 0);

    if (cache_clean_interval > UINT_MAX) {

        error_setg(errp, "Cache clean interval too big");

        ret = -EINVAL;

        goto fail;

    }



    /* Enable lazy_refcounts according to image and command line options */

    use_lazy_refcounts = qemu_opt_get_bool(opts, QCOW2_OPT_LAZY_REFCOUNTS,

        (s->compatible_features & QCOW2_COMPAT_LAZY_REFCOUNTS));

    if (use_lazy_refcounts && s->qcow_version < 3) {

        error_setg(errp, "Lazy refcounts require a qcow2 image with at least "

                   "qemu 1.1 compatibility level");

        ret = -EINVAL;

        goto fail;

    }



    /* Overlap check options */

    opt_overlap_check = qemu_opt_get(opts, QCOW2_OPT_OVERLAP);

    opt_overlap_check_template = qemu_opt_get(opts, QCOW2_OPT_OVERLAP_TEMPLATE);

    if (opt_overlap_check_template && opt_overlap_check &&

        strcmp(opt_overlap_check_template, opt_overlap_check))

    {

        error_setg(errp, "Conflicting values for qcow2 options '"

                   QCOW2_OPT_OVERLAP "' ('%s') and '" QCOW2_OPT_OVERLAP_TEMPLATE

                   "' ('%s')", opt_overlap_check, opt_overlap_check_template);

        ret = -EINVAL;

        goto fail;

    }

    if (!opt_overlap_check) {

        opt_overlap_check = opt_overlap_check_template ?: "cached";

    }



    if (!strcmp(opt_overlap_check, "none")) {

        overlap_check_template = 0;

    } else if (!strcmp(opt_overlap_check, "constant")) {

        overlap_check_template = QCOW2_OL_CONSTANT;

    } else if (!strcmp(opt_overlap_check, "cached")) {

        overlap_check_template = QCOW2_OL_CACHED;

    } else if (!strcmp(opt_overlap_check, "all")) {

        overlap_check_template = QCOW2_OL_ALL;

    } else {

        error_setg(errp, "Unsupported value '%s' for qcow2 option "

                   "'overlap-check'. Allowed are any of the following: "

                   "none, constant, cached, all", opt_overlap_check);

        ret = -EINVAL;

        goto fail;

    }



    /*

     * Start updating fields in BDRVQcow2State.

     * After this point no failure is allowed any more.

     */

    s->overlap_check = 0;

    for (i = 0; i < QCOW2_OL_MAX_BITNR; i++) {

        /* overlap-check defines a template bitmask, but every flag may be

         * overwritten through the associated boolean option */

        s->overlap_check |=

            qemu_opt_get_bool(opts, overlap_bool_option_names[i],

                              overlap_check_template & (1 << i)) << i;

    }



    s->l2_table_cache = l2_table_cache;

    s->refcount_block_cache = refcount_block_cache;



    s->use_lazy_refcounts = use_lazy_refcounts;



    s->discard_passthrough[QCOW2_DISCARD_NEVER] = false;

    s->discard_passthrough[QCOW2_DISCARD_ALWAYS] = true;

    s->discard_passthrough[QCOW2_DISCARD_REQUEST] =

        qemu_opt_get_bool(opts, QCOW2_OPT_DISCARD_REQUEST,

                          flags & BDRV_O_UNMAP);

    s->discard_passthrough[QCOW2_DISCARD_SNAPSHOT] =

        qemu_opt_get_bool(opts, QCOW2_OPT_DISCARD_SNAPSHOT, true);

    s->discard_passthrough[QCOW2_DISCARD_OTHER] =

        qemu_opt_get_bool(opts, QCOW2_OPT_DISCARD_OTHER, false);



    s->cache_clean_interval = cache_clean_interval;

    cache_clean_timer_init(bs, bdrv_get_aio_context(bs));



    ret = 0;

fail:

    qemu_opts_del(opts);

    opts = NULL;



    return ret;

}
