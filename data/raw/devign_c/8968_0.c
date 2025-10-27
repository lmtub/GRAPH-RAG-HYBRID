static BlockBackend *bdrv_first_blk(BlockDriverState *bs)

{

    BdrvChild *child;

    QLIST_FOREACH(child, &bs->parents, next_parent) {

        if (child->role == &child_root) {

            assert(bs->blk);

            return child->opaque;

        }

    }



    assert(!bs->blk);

    return NULL;

}
