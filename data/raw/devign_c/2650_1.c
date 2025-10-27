static void mirror_start_job(const char *job_id, BlockDriverState *bs,

                             int creation_flags, BlockDriverState *target,

                             const char *replaces, int64_t speed,

                             uint32_t granularity, int64_t buf_size,

                             BlockMirrorBackingMode backing_mode,

                             BlockdevOnError on_source_error,

                             BlockdevOnError on_target_error,

                             bool unmap,

                             BlockCompletionFunc *cb,

                             void *opaque, Error **errp,

                             const BlockJobDriver *driver,

                             bool is_none_mode, BlockDriverState *base,

                             bool auto_complete, const char *filter_node_name)

{

    MirrorBlockJob *s;

    BlockDriverState *mirror_top_bs;

    bool target_graph_mod;

    bool target_is_backing;

    Error *local_err = NULL;

    int ret;



    if (granularity == 0) {

        granularity = bdrv_get_default_bitmap_granularity(target);

    }



    assert ((granularity & (granularity - 1)) == 0);



    if (buf_size < 0) {

        error_setg(errp, "Invalid parameter 'buf-size'");

        return;

    }



    if (buf_size == 0) {

        buf_size = DEFAULT_MIRROR_BUF_SIZE;

    }



    /* In the case of active commit, add dummy driver to provide consistent

     * reads on the top, while disabling it in the intermediate nodes, and make

     * the backing chain writable. */

    mirror_top_bs = bdrv_new_open_driver(&bdrv_mirror_top, filter_node_name,

                                         BDRV_O_RDWR, errp);

    if (mirror_top_bs == NULL) {

        return;

    }

    mirror_top_bs->total_sectors = bs->total_sectors;



    /* bdrv_append takes ownership of the mirror_top_bs reference, need to keep

     * it alive until block_job_create() even if bs has no parent. */

    bdrv_ref(mirror_top_bs);

    bdrv_drained_begin(bs);

    bdrv_append(mirror_top_bs, bs, &local_err);

    bdrv_drained_end(bs);



    if (local_err) {

        bdrv_unref(mirror_top_bs);

        error_propagate(errp, local_err);

        return;

    }



    /* Make sure that the source is not resized while the job is running */

    s = block_job_create(job_id, driver, mirror_top_bs,

                         BLK_PERM_CONSISTENT_READ,

                         BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED |

                         BLK_PERM_WRITE | BLK_PERM_GRAPH_MOD, speed,

                         creation_flags, cb, opaque, errp);

    bdrv_unref(mirror_top_bs);

    if (!s) {

        goto fail;

    }

    s->source = bs;

    s->mirror_top_bs = mirror_top_bs;



    /* No resize for the target either; while the mirror is still running, a

     * consistent read isn't necessarily possible. We could possibly allow

     * writes and graph modifications, though it would likely defeat the

     * purpose of a mirror, so leave them blocked for now.

     *

     * In the case of active commit, things look a bit different, though,

     * because the target is an already populated backing file in active use.

     * We can allow anything except resize there.*/

    target_is_backing = bdrv_chain_contains(bs, target);

    target_graph_mod = (backing_mode != MIRROR_LEAVE_BACKING_CHAIN);

    s->target = blk_new(BLK_PERM_WRITE | BLK_PERM_RESIZE |

                        (target_graph_mod ? BLK_PERM_GRAPH_MOD : 0),

                        BLK_PERM_WRITE_UNCHANGED |

                        (target_is_backing ? BLK_PERM_CONSISTENT_READ |

                                             BLK_PERM_WRITE |

                                             BLK_PERM_GRAPH_MOD : 0));

    ret = blk_insert_bs(s->target, target, errp);

    if (ret < 0) {

        goto fail;

    }



    s->replaces = g_strdup(replaces);

    s->on_source_error = on_source_error;

    s->on_target_error = on_target_error;

    s->is_none_mode = is_none_mode;

    s->backing_mode = backing_mode;

    s->base = base;

    s->granularity = granularity;

    s->buf_size = ROUND_UP(buf_size, granularity);

    s->unmap = unmap;

    if (auto_complete) {

        s->should_complete = true;

    }



    s->dirty_bitmap = bdrv_create_dirty_bitmap(bs, granularity, NULL, errp);

    if (!s->dirty_bitmap) {

        goto fail;

    }



    /* Required permissions are already taken with blk_new() */

    block_job_add_bdrv(&s->common, "target", target, 0, BLK_PERM_ALL,

                       &error_abort);



    /* In commit_active_start() all intermediate nodes disappear, so

     * any jobs in them must be blocked */

    if (target_is_backing) {

        BlockDriverState *iter;

        for (iter = backing_bs(bs); iter != target; iter = backing_bs(iter)) {

            /* XXX BLK_PERM_WRITE needs to be allowed so we don't block

             * ourselves at s->base (if writes are blocked for a node, they are

             * also blocked for its backing file). The other options would be a

             * second filter driver above s->base (== target). */

            ret = block_job_add_bdrv(&s->common, "intermediate node", iter, 0,

                                     BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE,

                                     errp);

            if (ret < 0) {

                goto fail;

            }

        }

    }



    trace_mirror_start(bs, s, opaque);

    block_job_start(&s->common);

    return;



fail:

    if (s) {

        g_free(s->replaces);

        blk_unref(s->target);

        block_job_unref(&s->common);

    }



    bdrv_child_try_set_perm(mirror_top_bs->backing, 0, BLK_PERM_ALL,

                            &error_abort);

    bdrv_replace_node(mirror_top_bs, backing_bs(mirror_top_bs), &error_abort);

}
