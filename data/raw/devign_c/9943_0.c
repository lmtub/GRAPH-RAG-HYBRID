void qmp_transaction(TransactionActionList *dev_list, Error **errp)

{

    TransactionActionList *dev_entry = dev_list;

    BlkTransactionState *state, *next;

    Error *local_err = NULL;



    QSIMPLEQ_HEAD(snap_bdrv_states, BlkTransactionState) snap_bdrv_states;

    QSIMPLEQ_INIT(&snap_bdrv_states);



    /* drain all i/o before any operations */

    bdrv_drain_all();



    /* We don't do anything in this loop that commits us to the operations */

    while (NULL != dev_entry) {

        TransactionAction *dev_info = NULL;

        const BdrvActionOps *ops;



        dev_info = dev_entry->value;

        dev_entry = dev_entry->next;



        assert(dev_info->kind < ARRAY_SIZE(actions));



        ops = &actions[dev_info->kind];

        assert(ops->instance_size > 0);



        state = g_malloc0(ops->instance_size);

        state->ops = ops;

        state->action = dev_info;

        QSIMPLEQ_INSERT_TAIL(&snap_bdrv_states, state, entry);



        state->ops->prepare(state, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            goto delete_and_fail;

        }

    }



    QSIMPLEQ_FOREACH(state, &snap_bdrv_states, entry) {

        if (state->ops->commit) {

            state->ops->commit(state);

        }

    }



    /* success */

    goto exit;



delete_and_fail:

    /* failure, and it is all-or-none; roll back all operations */

    QSIMPLEQ_FOREACH(state, &snap_bdrv_states, entry) {

        if (state->ops->abort) {

            state->ops->abort(state);

        }

    }

exit:

    QSIMPLEQ_FOREACH_SAFE(state, &snap_bdrv_states, entry, next) {

        if (state->ops->clean) {

            state->ops->clean(state);

        }

        g_free(state);

    }

}
