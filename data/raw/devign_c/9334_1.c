void qmp_migrate_set_capabilities(MigrationCapabilityStatusList *params,

                                  Error **errp)

{

    MigrationState *s = migrate_get_current();

    MigrationCapabilityStatusList *cap;



    if (s->state == MIG_STATE_ACTIVE || s->state == MIG_STATE_SETUP) {

        error_set(errp, QERR_MIGRATION_ACTIVE);

        return;

    }



    for (cap = params; cap; cap = cap->next) {

        s->enabled_capabilities[cap->value->capability] = cap->value->state;

    }

}
