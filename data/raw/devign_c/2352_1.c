void stream_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *base, const char *backing_file_str,

                  int64_t speed, BlockdevOnError on_error, Error **errp)

{

    StreamBlockJob *s;

    BlockDriverState *iter;

    int orig_bs_flags;



    /* Make sure that the image is opened in read-write mode */

    orig_bs_flags = bdrv_get_flags(bs);

    if (!(orig_bs_flags & BDRV_O_RDWR)) {

        if (bdrv_reopen(bs, orig_bs_flags | BDRV_O_RDWR, errp) != 0) {

            return;

        }

    }



    /* Prevent concurrent jobs trying to modify the graph structure here, we

     * already have our own plans. Also don't allow resize as the image size is

     * queried only at the job start and then cached. */

    s = block_job_create(job_id, &stream_job_driver, bs,

                         BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED |

                         BLK_PERM_GRAPH_MOD,

                         BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED |

                         BLK_PERM_WRITE,

                         speed, BLOCK_JOB_DEFAULT, NULL, NULL, errp);

    if (!s) {

        goto fail;

    }



    /* Block all intermediate nodes between bs and base, because they will

     * disappear from the chain after this operation. The streaming job reads

     * every block only once, assuming that it doesn't change, so block writes

     * and resizes. */

    for (iter = backing_bs(bs); iter && iter != base; iter = backing_bs(iter)) {

        block_job_add_bdrv(&s->common, "intermediate node", iter, 0,

                           BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED,

                           &error_abort);

    }



    s->base = base;

    s->backing_file_str = g_strdup(backing_file_str);

    s->bs_flags = orig_bs_flags;



    s->on_error = on_error;

    trace_stream_start(bs, base, s);

    block_job_start(&s->common);

    return;



fail:

    if (orig_bs_flags != bdrv_get_flags(bs)) {

        bdrv_reopen(bs, s->bs_flags, NULL);

    }

}
