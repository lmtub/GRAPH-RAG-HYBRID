BdrvChild *bdrv_root_attach_child(BlockDriverState *child_bs,

                                  const char *child_name,

                                  const BdrvChildRole *child_role,

                                  uint64_t perm, uint64_t shared_perm,

                                  void *opaque, Error **errp)

{

    BdrvChild *child;

    int ret;



    ret = bdrv_check_update_perm(child_bs, perm, shared_perm, NULL, errp);

    if (ret < 0) {

        return NULL;

    }



    child = g_new(BdrvChild, 1);

    *child = (BdrvChild) {

        .bs             = NULL,

        .name           = g_strdup(child_name),

        .role           = child_role,

        .perm           = perm,

        .shared_perm    = shared_perm,

        .opaque         = opaque,

    };



    bdrv_replace_child(child, child_bs);



    return child;

}
