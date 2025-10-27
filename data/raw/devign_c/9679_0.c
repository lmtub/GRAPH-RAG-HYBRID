static int qcow2_create2(const char *filename, int64_t total_size,

                         const char *backing_file, const char *backing_format,

                         int flags, size_t cluster_size, PreallocMode prealloc,

                         QemuOpts *opts, int version, int refcount_order,

                         const char *encryptfmt, Error **errp)

{

    int cluster_bits;

    QDict *options;



    /* Calculate cluster_bits */

    cluster_bits = ctz32(cluster_size);

    if (cluster_bits < MIN_CLUSTER_BITS || cluster_bits > MAX_CLUSTER_BITS ||

        (1 << cluster_bits) != cluster_size)

    {

        error_setg(errp, "Cluster size must be a power of two between %d and "

                   "%dk", 1 << MIN_CLUSTER_BITS, 1 << (MAX_CLUSTER_BITS - 10));

        return -EINVAL;

    }



    /*

     * Open the image file and write a minimal qcow2 header.

     *

     * We keep things simple and start with a zero-sized image. We also

     * do without refcount blocks or a L1 table for now. We'll fix the

     * inconsistency later.

     *

     * We do need a refcount table because growing the refcount table means

     * allocating two new refcount blocks - the seconds of which would be at

     * 2 GB for 64k clusters, and we don't want to have a 2 GB initial file

     * size for any qcow2 image.

     */

    BlockBackend *blk;

    QCowHeader *header;

    uint64_t* refcount_table;

    Error *local_err = NULL;

    int ret;



    if (prealloc == PREALLOC_MODE_FULL || prealloc == PREALLOC_MODE_FALLOC) {

        int64_t prealloc_size =

            qcow2_calc_prealloc_size(total_size, cluster_size, refcount_order);

        qemu_opt_set_number(opts, BLOCK_OPT_SIZE, prealloc_size, &error_abort);

        qemu_opt_set(opts, BLOCK_OPT_PREALLOC, PreallocMode_lookup[prealloc],

                     &error_abort);

    }



    ret = bdrv_create_file(filename, opts, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        return ret;

    }



    blk = blk_new_open(filename, NULL, NULL,

                       BDRV_O_RDWR | BDRV_O_RESIZE | BDRV_O_PROTOCOL,

                       &local_err);

    if (blk == NULL) {

        error_propagate(errp, local_err);

        return -EIO;

    }



    blk_set_allow_write_beyond_eof(blk, true);



    /* Write the header */

    QEMU_BUILD_BUG_ON((1 << MIN_CLUSTER_BITS) < sizeof(*header));

    header = g_malloc0(cluster_size);

    *header = (QCowHeader) {

        .magic                      = cpu_to_be32(QCOW_MAGIC),

        .version                    = cpu_to_be32(version),

        .cluster_bits               = cpu_to_be32(cluster_bits),

        .size                       = cpu_to_be64(0),

        .l1_table_offset            = cpu_to_be64(0),

        .l1_size                    = cpu_to_be32(0),

        .refcount_table_offset      = cpu_to_be64(cluster_size),

        .refcount_table_clusters    = cpu_to_be32(1),

        .refcount_order             = cpu_to_be32(refcount_order),

        .header_length              = cpu_to_be32(sizeof(*header)),

    };



    /* We'll update this to correct value later */

    header->crypt_method = cpu_to_be32(QCOW_CRYPT_NONE);



    if (flags & BLOCK_FLAG_LAZY_REFCOUNTS) {

        header->compatible_features |=

            cpu_to_be64(QCOW2_COMPAT_LAZY_REFCOUNTS);

    }



    ret = blk_pwrite(blk, 0, header, cluster_size, 0);

    g_free(header);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write qcow2 header");

        goto out;

    }



    /* Write a refcount table with one refcount block */

    refcount_table = g_malloc0(2 * cluster_size);

    refcount_table[0] = cpu_to_be64(2 * cluster_size);

    ret = blk_pwrite(blk, cluster_size, refcount_table, 2 * cluster_size, 0);

    g_free(refcount_table);



    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write refcount table");

        goto out;

    }



    blk_unref(blk);

    blk = NULL;



    /*

     * And now open the image and make it consistent first (i.e. increase the

     * refcount of the cluster that is occupied by the header and the refcount

     * table)

     */

    options = qdict_new();

    qdict_put_str(options, "driver", "qcow2");

    blk = blk_new_open(filename, NULL, options,

                       BDRV_O_RDWR | BDRV_O_RESIZE | BDRV_O_NO_FLUSH,

                       &local_err);

    if (blk == NULL) {

        error_propagate(errp, local_err);

        ret = -EIO;

        goto out;

    }



    ret = qcow2_alloc_clusters(blk_bs(blk), 3 * cluster_size);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not allocate clusters for qcow2 "

                         "header and refcount table");

        goto out;



    } else if (ret != 0) {

        error_report("Huh, first cluster in empty image is already in use?");

        abort();

    }



    /* Create a full header (including things like feature table) */

    ret = qcow2_update_header(blk_bs(blk));

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not update qcow2 header");

        goto out;

    }



    /* Okay, now that we have a valid image, let's give it the right size */

    ret = blk_truncate(blk, total_size, errp);

    if (ret < 0) {

        error_prepend(errp, "Could not resize image: ");

        goto out;

    }



    /* Want a backing file? There you go.*/

    if (backing_file) {

        ret = bdrv_change_backing_file(blk_bs(blk), backing_file, backing_format);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Could not assign backing file '%s' "

                             "with format '%s'", backing_file, backing_format);

            goto out;

        }

    }



    /* Want encryption? There you go. */

    if (encryptfmt) {

        ret = qcow2_set_up_encryption(blk_bs(blk), encryptfmt, opts, errp);

        if (ret < 0) {

            goto out;

        }

    }



    /* And if we're supposed to preallocate metadata, do that now */

    if (prealloc != PREALLOC_MODE_OFF) {

        BDRVQcow2State *s = blk_bs(blk)->opaque;

        qemu_co_mutex_lock(&s->lock);

        ret = preallocate(blk_bs(blk));

        qemu_co_mutex_unlock(&s->lock);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Could not preallocate metadata");

            goto out;

        }

    }



    blk_unref(blk);

    blk = NULL;



    /* Reopen the image without BDRV_O_NO_FLUSH to flush it before returning.

     * Using BDRV_O_NO_IO, since encryption is now setup we don't want to

     * have to setup decryption context. We're not doing any I/O on the top

     * level BlockDriverState, only lower layers, where BDRV_O_NO_IO does

     * not have effect.

     */

    options = qdict_new();

    qdict_put_str(options, "driver", "qcow2");

    blk = blk_new_open(filename, NULL, options,

                       BDRV_O_RDWR | BDRV_O_NO_BACKING | BDRV_O_NO_IO,

                       &local_err);

    if (blk == NULL) {

        error_propagate(errp, local_err);

        ret = -EIO;

        goto out;

    }



    ret = 0;

out:

    if (blk) {

        blk_unref(blk);

    }

    return ret;

}
