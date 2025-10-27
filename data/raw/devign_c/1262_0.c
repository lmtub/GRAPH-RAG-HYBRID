static int bdrv_check_perm(BlockDriverState *bs, uint64_t cumulative_perms,

                           uint64_t cumulative_shared_perms, Error **errp)

{

    BlockDriver *drv = bs->drv;

    BdrvChild *c;

    int ret;



    /* Write permissions never work with read-only images */

    if ((cumulative_perms & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED)) &&

        bdrv_is_read_only(bs))

    {

        error_setg(errp, "Block node is read-only");

        return -EPERM;

    }



    /* Check this node */

    if (!drv) {

        return 0;

    }



    if (drv->bdrv_check_perm) {

        return drv->bdrv_check_perm(bs, cumulative_perms,

                                    cumulative_shared_perms, errp);

    }



    /* Drivers that never have children can omit .bdrv_child_perm() */

    if (!drv->bdrv_child_perm) {

        assert(QLIST_EMPTY(&bs->children));

        return 0;

    }



    /* Check all children */

    QLIST_FOREACH(c, &bs->children, next) {

        uint64_t cur_perm, cur_shared;

        drv->bdrv_child_perm(bs, c, c->role,

                             cumulative_perms, cumulative_shared_perms,

                             &cur_perm, &cur_shared);

        ret = bdrv_child_check_perm(c, cur_perm, cur_shared, errp);

        if (ret < 0) {

            return ret;

        }

    }



    return 0;

}
