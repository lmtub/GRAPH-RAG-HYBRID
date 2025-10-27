static void *rcu_q_updater(void *arg)

{

    int j, target_el;

    long long n_updates_local = 0;

    long long n_removed_local = 0;

    struct list_element *el, *prev_el;



    *(struct rcu_reader_data **)arg = &rcu_reader;

    atomic_inc(&nthreadsrunning);

    while (goflag == GOFLAG_INIT) {

        g_usleep(1000);

    }



    while (goflag == GOFLAG_RUN) {

        target_el = select_random_el(RCU_Q_LEN);

        j = 0;

        /* FOREACH_RCU could work here but let's use both macros */

        QLIST_FOREACH_SAFE_RCU(prev_el, &Q_list_head, entry, el) {

            j++;

            if (target_el == j) {

                QLIST_REMOVE_RCU(prev_el, entry);

                /* may be more than one updater in the future */

                call_rcu1(&prev_el->rcu, reclaim_list_el);

                n_removed_local++;

                break;

            }

        }

        if (goflag == GOFLAG_STOP) {

            break;

        }

        target_el = select_random_el(RCU_Q_LEN);

        j = 0;

        QLIST_FOREACH_RCU(el, &Q_list_head, entry) {

            j++;

            if (target_el == j) {

                prev_el = g_new(struct list_element, 1);

                atomic_add(&n_nodes, 1);

                prev_el->val = atomic_read(&n_nodes);

                QLIST_INSERT_BEFORE_RCU(el, prev_el, entry);

                break;

            }

        }



        n_updates_local += 2;

        synchronize_rcu();

    }

    synchronize_rcu();

    atomic_add(&n_updates, n_updates_local);

    atomic_add(&n_nodes_removed, n_removed_local);

    return NULL;

}
