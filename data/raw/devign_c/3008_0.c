void do_info_migrate(Monitor *mon, QObject **ret_data)

{

    QDict *qdict;

    MigrationState *s = current_migration;



    if (s) {

        switch (s->get_status(s)) {

        case MIG_STATE_ACTIVE:

            qdict = qdict_new();

            qdict_put(qdict, "status", qstring_from_str("active"));



            migrate_put_status(qdict, "ram", ram_bytes_transferred(),

                               ram_bytes_remaining(), ram_bytes_total());



            if (blk_mig_active()) {

                migrate_put_status(qdict, "disk", blk_mig_bytes_transferred(),

                                   blk_mig_bytes_remaining(),

                                   blk_mig_bytes_total());

            }



            *ret_data = QOBJECT(qdict);

            break;

        case MIG_STATE_COMPLETED:

            *ret_data = qobject_from_jsonf("{ 'status': 'completed' }");

            break;

        case MIG_STATE_ERROR:

            *ret_data = qobject_from_jsonf("{ 'status': 'failed' }");

            break;

        case MIG_STATE_CANCELLED:

            *ret_data = qobject_from_jsonf("{ 'status': 'cancelled' }");

            break;

        }

        assert(*ret_data != NULL);

    }

}
