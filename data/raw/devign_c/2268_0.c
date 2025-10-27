BlockBackend *blk_new_with_bs(Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;



    blk = blk_new(errp);

    if (!blk) {

        return NULL;

    }



    bs = bdrv_new_root();

    blk->root = bdrv_root_attach_child(bs, "root", &child_root);

    blk->root->opaque = blk;

    bs->blk = blk;

    return blk;

}
