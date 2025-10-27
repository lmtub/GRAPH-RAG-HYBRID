static int qcow2_amend_options(BlockDriverState *bs, QemuOpts *opts,

                               BlockDriverAmendStatusCB *status_cb,

                               void *cb_opaque)

{

    BDRVQcow2State *s = bs->opaque;

    int old_version = s->qcow_version, new_version = old_version;

    uint64_t new_size = 0;

    const char *backing_file = NULL, *backing_format = NULL;

    bool lazy_refcounts = s->use_lazy_refcounts;

    const char *compat = NULL;

    uint64_t cluster_size = s->cluster_size;

    bool encrypt;

    int refcount_bits = s->refcount_bits;

    Error *local_err = NULL;

    int ret;

    QemuOptDesc *desc = opts->list->desc;

    Qcow2AmendHelperCBInfo helper_cb_info;



    while (desc && desc->name) {

        if (!qemu_opt_find(opts, desc->name)) {

            /* only change explicitly defined options */

            desc++;

            continue;

        }



        if (!strcmp(desc->name, BLOCK_OPT_COMPAT_LEVEL)) {

            compat = qemu_opt_get(opts, BLOCK_OPT_COMPAT_LEVEL);

            if (!compat) {

                /* preserve default */

            } else if (!strcmp(compat, "0.10")) {

                new_version = 2;

            } else if (!strcmp(compat, "1.1")) {

                new_version = 3;

            } else {

                error_report("Unknown compatibility level %s", compat);

                return -EINVAL;

            }

        } else if (!strcmp(desc->name, BLOCK_OPT_PREALLOC)) {

            error_report("Cannot change preallocation mode");

            return -ENOTSUP;

        } else if (!strcmp(desc->name, BLOCK_OPT_SIZE)) {

            new_size = qemu_opt_get_size(opts, BLOCK_OPT_SIZE, 0);

        } else if (!strcmp(desc->name, BLOCK_OPT_BACKING_FILE)) {

            backing_file = qemu_opt_get(opts, BLOCK_OPT_BACKING_FILE);

        } else if (!strcmp(desc->name, BLOCK_OPT_BACKING_FMT)) {

            backing_format = qemu_opt_get(opts, BLOCK_OPT_BACKING_FMT);

        } else if (!strcmp(desc->name, BLOCK_OPT_ENCRYPT)) {

            encrypt = qemu_opt_get_bool(opts, BLOCK_OPT_ENCRYPT,

                                        !!s->cipher);



            if (encrypt != !!s->cipher) {

                error_report("Changing the encryption flag is not supported");

                return -ENOTSUP;

            }

        } else if (!strcmp(desc->name, BLOCK_OPT_CLUSTER_SIZE)) {

            cluster_size = qemu_opt_get_size(opts, BLOCK_OPT_CLUSTER_SIZE,

                                             cluster_size);

            if (cluster_size != s->cluster_size) {

                error_report("Changing the cluster size is not supported");

                return -ENOTSUP;

            }

        } else if (!strcmp(desc->name, BLOCK_OPT_LAZY_REFCOUNTS)) {

            lazy_refcounts = qemu_opt_get_bool(opts, BLOCK_OPT_LAZY_REFCOUNTS,

                                               lazy_refcounts);

        } else if (!strcmp(desc->name, BLOCK_OPT_REFCOUNT_BITS)) {

            refcount_bits = qemu_opt_get_number(opts, BLOCK_OPT_REFCOUNT_BITS,

                                                refcount_bits);



            if (refcount_bits <= 0 || refcount_bits > 64 ||

                !is_power_of_2(refcount_bits))

            {

                error_report("Refcount width must be a power of two and may "

                             "not exceed 64 bits");

                return -EINVAL;

            }

        } else {

            /* if this point is reached, this probably means a new option was

             * added without having it covered here */

            abort();

        }



        desc++;

    }



    helper_cb_info = (Qcow2AmendHelperCBInfo){

        .original_status_cb = status_cb,

        .original_cb_opaque = cb_opaque,

        .total_operations = (new_version < old_version)

                          + (s->refcount_bits != refcount_bits)

    };



    /* Upgrade first (some features may require compat=1.1) */

    if (new_version > old_version) {

        s->qcow_version = new_version;

        ret = qcow2_update_header(bs);

        if (ret < 0) {

            s->qcow_version = old_version;

            return ret;

        }

    }



    if (s->refcount_bits != refcount_bits) {

        int refcount_order = ctz32(refcount_bits);



        if (new_version < 3 && refcount_bits != 16) {

            error_report("Different refcount widths than 16 bits require "

                         "compatibility level 1.1 or above (use compat=1.1 or "

                         "greater)");

            return -EINVAL;

        }



        helper_cb_info.current_operation = QCOW2_CHANGING_REFCOUNT_ORDER;

        ret = qcow2_change_refcount_order(bs, refcount_order,

                                          &qcow2_amend_helper_cb,

                                          &helper_cb_info, &local_err);

        if (ret < 0) {

            error_report_err(local_err);

            return ret;

        }

    }



    if (backing_file || backing_format) {

        ret = qcow2_change_backing_file(bs,

                    backing_file ?: s->image_backing_file,

                    backing_format ?: s->image_backing_format);

        if (ret < 0) {

            return ret;

        }

    }



    if (s->use_lazy_refcounts != lazy_refcounts) {

        if (lazy_refcounts) {

            if (new_version < 3) {

                error_report("Lazy refcounts only supported with compatibility "

                             "level 1.1 and above (use compat=1.1 or greater)");

                return -EINVAL;

            }

            s->compatible_features |= QCOW2_COMPAT_LAZY_REFCOUNTS;

            ret = qcow2_update_header(bs);

            if (ret < 0) {

                s->compatible_features &= ~QCOW2_COMPAT_LAZY_REFCOUNTS;

                return ret;

            }

            s->use_lazy_refcounts = true;

        } else {

            /* make image clean first */

            ret = qcow2_mark_clean(bs);

            if (ret < 0) {

                return ret;

            }

            /* now disallow lazy refcounts */

            s->compatible_features &= ~QCOW2_COMPAT_LAZY_REFCOUNTS;

            ret = qcow2_update_header(bs);

            if (ret < 0) {

                s->compatible_features |= QCOW2_COMPAT_LAZY_REFCOUNTS;

                return ret;

            }

            s->use_lazy_refcounts = false;

        }

    }



    if (new_size) {

        BlockBackend *blk = blk_new(BLK_PERM_RESIZE, BLK_PERM_ALL);

        ret = blk_insert_bs(blk, bs, &local_err);

        if (ret < 0) {

            error_report_err(local_err);

            blk_unref(blk);

            return ret;

        }



        ret = blk_truncate(blk, new_size, &local_err);

        blk_unref(blk);

        if (ret < 0) {

            error_report_err(local_err);

            return ret;

        }

    }



    /* Downgrade last (so unsupported features can be removed before) */

    if (new_version < old_version) {

        helper_cb_info.current_operation = QCOW2_DOWNGRADING;

        ret = qcow2_downgrade(bs, new_version, &qcow2_amend_helper_cb,

                              &helper_cb_info);

        if (ret < 0) {

            return ret;

        }

    }



    return 0;

}
