static void change_parent_backing_link(BlockDriverState *from,

                                       BlockDriverState *to)

{

    BdrvChild *c, *next, *to_c;



    QLIST_FOREACH_SAFE(c, &from->parents, next_parent, next) {

        if (c->role->stay_at_node) {

            continue;

        }

        if (c->role == &child_backing) {

            /* @from is generally not allowed to be a backing file, except for

             * when @to is the overlay. In that case, @from may not be replaced

             * by @to as @to's backing node. */

            QLIST_FOREACH(to_c, &to->children, next) {

                if (to_c == c) {

                    break;

                }

            }

            if (to_c) {

                continue;

            }

        }



        assert(c->role != &child_backing);

        bdrv_ref(to);

        /* FIXME Are we sure that bdrv_replace_child() can't run into

         * &error_abort because of permissions? */

        bdrv_replace_child(c, to, true);

        bdrv_unref(from);

    }

}
