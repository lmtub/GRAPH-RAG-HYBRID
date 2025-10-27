static void rcu_qtest_init(void)

{

    struct list_element *new_el;

    int i;

    nthreadsrunning = 0;

    srand(time(0));

    for (i = 0; i < RCU_Q_LEN; i++) {

        new_el = g_new(struct list_element, 1);

        new_el->val = i;

        QLIST_INSERT_HEAD_RCU(&Q_list_head, new_el, entry);

    }

    atomic_add(&n_nodes, RCU_Q_LEN);

}
