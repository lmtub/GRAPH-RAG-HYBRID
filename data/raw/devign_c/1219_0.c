int bdrv_child_check_perm(BdrvChild *c, uint64_t perm, uint64_t shared,

                          Error **errp)

{

    return bdrv_check_update_perm(c->bs, perm, shared, c, errp);

}
