void commit_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *base, BlockDriverState *top, int64_t speed,

                  BlockdevOnError on_error, const char *backing_file_str,

                  const char *filter_node_name, Error **errp)

{

    CommitBlockJob *s;

    BlockReopenQueue *reopen_queue = NULL;

    int orig_overlay_flags;

    int orig_base_flags;

    BlockDriverState *iter;

    BlockDriverState *overlay_bs;

    BlockDriverState *commit_top_bs = NULL;

    Error *local_err = NULL;

    int ret;



    assert(top != bs);

    if (top == base) {

        error_setg(errp, "Invalid files for merge: top and base are the same");

        return;




    overlay_bs = bdrv_find_overlay(bs, top);



    if (overlay_bs == NULL) {

        error_setg(errp, "Could not find overlay image for %s:", top->filename);

        return;




    s = block_job_create(job_id, &commit_job_driver, bs, 0, BLK_PERM_ALL,

                         speed, BLOCK_JOB_DEFAULT, NULL, NULL, errp);

    if (!s) {

        return;




    orig_base_flags    = bdrv_get_flags(base);

    orig_overlay_flags = bdrv_get_flags(overlay_bs);



    /* convert base & overlay_bs to r/w, if necessary */

    if (!(orig_base_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, base, NULL,

                                         orig_base_flags | BDRV_O_RDWR);


    if (!(orig_overlay_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, overlay_bs, NULL,

                                         orig_overlay_flags | BDRV_O_RDWR);


    if (reopen_queue) {

        bdrv_reopen_multiple(bdrv_get_aio_context(bs), reopen_queue, &local_err);

        if (local_err != NULL) {

            error_propagate(errp, local_err);

            goto fail;





    /* Insert commit_top block node above top, so we can block consistent read

     * on the backing chain below it */

    commit_top_bs = bdrv_new_open_driver(&bdrv_commit_top, filter_node_name, 0,

                                         errp);

    if (commit_top_bs == NULL) {

        goto fail;





    commit_top_bs->total_sectors = top->total_sectors;

    bdrv_set_aio_context(commit_top_bs, bdrv_get_aio_context(top));



    bdrv_set_backing_hd(commit_top_bs, top, &local_err);

    if (local_err) {

        bdrv_unref(commit_top_bs);

        commit_top_bs = NULL;

        error_propagate(errp, local_err);

        goto fail;


    bdrv_set_backing_hd(overlay_bs, commit_top_bs, &local_err);

    if (local_err) {

        bdrv_unref(commit_top_bs);

        commit_top_bs = NULL;

        error_propagate(errp, local_err);

        goto fail;




    s->commit_top_bs = commit_top_bs;

    bdrv_unref(commit_top_bs);



    /* Block all nodes between top and base, because they will

     * disappear from the chain after this operation. */

    assert(bdrv_chain_contains(top, base));

    for (iter = top; iter != base; iter = backing_bs(iter)) {

        /* XXX BLK_PERM_WRITE needs to be allowed so we don't block ourselves

         * at s->base (if writes are blocked for a node, they are also blocked

         * for its backing file). The other options would be a second filter

         * driver above s->base. */

        ret = block_job_add_bdrv(&s->common, "intermediate node", iter, 0,

                                 BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE,

                                 errp);

        if (ret < 0) {

            goto fail;





    ret = block_job_add_bdrv(&s->common, "base", base, 0, BLK_PERM_ALL, errp);

    if (ret < 0) {

        goto fail;




    /* overlay_bs must be blocked because it needs to be modified to

     * update the backing image string. */

    ret = block_job_add_bdrv(&s->common, "overlay of top", overlay_bs,

                             BLK_PERM_GRAPH_MOD, BLK_PERM_ALL, errp);

    if (ret < 0) {

        goto fail;




    s->base = blk_new(BLK_PERM_CONSISTENT_READ

                      | BLK_PERM_WRITE

                      | BLK_PERM_RESIZE,

                      BLK_PERM_CONSISTENT_READ

                      | BLK_PERM_GRAPH_MOD

                      | BLK_PERM_WRITE_UNCHANGED);

    ret = blk_insert_bs(s->base, base, errp);

    if (ret < 0) {

        goto fail;




    /* Required permissions are already taken with block_job_add_bdrv() */

    s->top = blk_new(0, BLK_PERM_ALL);

    ret = blk_insert_bs(s->top, top, errp);

    if (ret < 0) {

        goto fail;




    s->active = bs;



    s->base_flags          = orig_base_flags;

    s->orig_overlay_flags  = orig_overlay_flags;



    s->backing_file_str = g_strdup(backing_file_str);



    s->on_error = on_error;



    trace_commit_start(bs, base, top, s);

    block_job_start(&s->common);

    return;



fail:

    if (s->base) {

        blk_unref(s->base);


    if (s->top) {

        blk_unref(s->top);


    if (commit_top_bs) {

        bdrv_set_backing_hd(overlay_bs, top, &error_abort);


    block_job_early_fail(&s->common);
