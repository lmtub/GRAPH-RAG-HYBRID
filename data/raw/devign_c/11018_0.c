static void migration_state_notifier(Notifier *notifier, void *data)

{

    MigrationState *s = data;



    if (migration_is_active(s)) {

#ifdef SPICE_INTERFACE_MIGRATION

        spice_server_migrate_start(spice_server);

#endif

    } else if (migration_has_finished(s)) {

#if SPICE_SERVER_VERSION >= 0x000701 /* 0.7.1 */

#ifndef SPICE_INTERFACE_MIGRATION

        spice_server_migrate_switch(spice_server);

#else

        spice_server_migrate_end(spice_server, true);

    } else if (migration_has_failed(s)) {

        spice_server_migrate_end(spice_server, false);

#endif

#endif

    }

}
