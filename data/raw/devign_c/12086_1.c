static void enter_migration_coroutine(void *opaque)

{

    Coroutine *co = opaque;

    qemu_coroutine_enter(co, NULL);

}
