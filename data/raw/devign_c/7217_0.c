static void bdrv_replace_child(BdrvChild *child, BlockDriverState *new_bs)

{

    BlockDriverState *old_bs = child->bs;



    if (old_bs) {

        if (old_bs->quiesce_counter && child->role->drained_end) {

            child->role->drained_end(child);

        }

        QLIST_REMOVE(child, next_parent);

    }



    child->bs = new_bs;



    if (new_bs) {

        QLIST_INSERT_HEAD(&new_bs->parents, child, next_parent);

        if (new_bs->quiesce_counter && child->role->drained_begin) {

            child->role->drained_begin(child);

        }

    }

}
