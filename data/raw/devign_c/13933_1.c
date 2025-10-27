static void migrate_finish_set_state(MigrationState *s, int new_state)

{

    if (atomic_cmpxchg(&s->state, MIG_STATE_ACTIVE, new_state) == new_state) {

        trace_migrate_set_state(new_state);

    }

}
