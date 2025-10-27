static void address_space_update_topology_pass(AddressSpace *as,

                                               const FlatView *old_view,

                                               const FlatView *new_view,

                                               bool adding)

{

    unsigned iold, inew;

    FlatRange *frold, *frnew;



    /* Generate a symmetric difference of the old and new memory maps.

     * Kill ranges in the old map, and instantiate ranges in the new map.

     */

    iold = inew = 0;

    while (iold < old_view->nr || inew < new_view->nr) {

        if (iold < old_view->nr) {

            frold = &old_view->ranges[iold];

        } else {

            frold = NULL;

        }

        if (inew < new_view->nr) {

            frnew = &new_view->ranges[inew];

        } else {

            frnew = NULL;

        }



        if (frold

            && (!frnew

                || int128_lt(frold->addr.start, frnew->addr.start)

                || (int128_eq(frold->addr.start, frnew->addr.start)

                    && !flatrange_equal(frold, frnew)))) {

            /* In old but not in new, or in both but attributes changed. */



            if (!adding) {

                MEMORY_LISTENER_UPDATE_REGION(frold, as, Reverse, region_del);

            }



            ++iold;

        } else if (frold && frnew && flatrange_equal(frold, frnew)) {

            /* In both and unchanged (except logging may have changed) */



            if (adding) {

                MEMORY_LISTENER_UPDATE_REGION(frnew, as, Forward, region_nop);

                if (frold->dirty_log_mask && !frnew->dirty_log_mask) {

                    MEMORY_LISTENER_UPDATE_REGION(frnew, as, Reverse, log_stop);

                } else if (frnew->dirty_log_mask && !frold->dirty_log_mask) {

                    MEMORY_LISTENER_UPDATE_REGION(frnew, as, Forward, log_start);

                }

            }



            ++iold;

            ++inew;

        } else {

            /* In new */



            if (adding) {

                MEMORY_LISTENER_UPDATE_REGION(frnew, as, Forward, region_add);

            }



            ++inew;

        }

    }

}
