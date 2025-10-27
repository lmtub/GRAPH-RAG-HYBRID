static int bdrv_check_update_perm(BlockDriverState *bs, uint64_t new_used_perm,

                                  uint64_t new_shared_perm,

                                  BdrvChild *ignore_child, Error **errp)

{

    BdrvChild *c;

    uint64_t cumulative_perms = new_used_perm;

    uint64_t cumulative_shared_perms = new_shared_perm;



    /* There is no reason why anyone couldn't tolerate write_unchanged */

    assert(new_shared_perm & BLK_PERM_WRITE_UNCHANGED);



    QLIST_FOREACH(c, &bs->parents, next_parent) {

        if (c == ignore_child) {

            continue;

        }



        if ((new_used_perm & c->shared_perm) != new_used_perm) {

            char *user = bdrv_child_user_desc(c);

            char *perm_names = bdrv_perm_names(new_used_perm & ~c->shared_perm);

            error_setg(errp, "Conflicts with use by %s as '%s', which does not "

                             "allow '%s' on %s",

                       user, c->name, perm_names, bdrv_get_node_name(c->bs));

            g_free(user);

            g_free(perm_names);

            return -EPERM;

        }



        if ((c->perm & new_shared_perm) != c->perm) {

            char *user = bdrv_child_user_desc(c);

            char *perm_names = bdrv_perm_names(c->perm & ~new_shared_perm);

            error_setg(errp, "Conflicts with use by %s as '%s', which uses "

                             "'%s' on %s",

                       user, c->name, perm_names, bdrv_get_node_name(c->bs));

            g_free(user);

            g_free(perm_names);

            return -EPERM;

        }



        cumulative_perms |= c->perm;

        cumulative_shared_perms &= c->shared_perm;

    }



    return bdrv_check_perm(bs, cumulative_perms, cumulative_shared_perms, errp);

}
