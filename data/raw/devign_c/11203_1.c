void migration_incoming_state_destroy(void)
{
    struct MigrationIncomingState *mis = migration_incoming_get_current();
    qemu_event_destroy(&mis->main_thread_load_event);