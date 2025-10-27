static int do_cont(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    struct bdrv_iterate_context context = { mon, 0 };



    if (incoming_expected) {

        qerror_report(QERR_MIGRATION_EXPECTED);

        return -1;

    }

    bdrv_iterate(encrypted_bdrv_it, &context);

    /* only resume the vm if all keys are set and valid */

    if (!context.err) {

        vm_start();

        return 0;

    } else {

        return -1;

    }

}
