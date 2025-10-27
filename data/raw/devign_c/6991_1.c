static void migrate_set_state(MigrationState *s, int old_state, int new_state)

{

    if (atomic_cmpxchg(&s->state, old_state, new_state) == new_state) {

        trace_migrate_set_state(new_state);

    }

}
