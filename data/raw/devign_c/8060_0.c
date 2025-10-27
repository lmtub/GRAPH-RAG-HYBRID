static bool migration_object_check(MigrationState *ms, Error **errp)

{

    if (!migrate_params_check(&ms->parameters, errp)) {

        return false;

    }



    return true;

}
