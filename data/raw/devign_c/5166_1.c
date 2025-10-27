bool migration_has_failed(MigrationState *s)

{

    return (s->state == MIG_STATE_CANCELLED ||

            s->state == MIG_STATE_ERROR);

}
