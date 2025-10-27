void migration_fd_process_incoming(QEMUFile *f)

{

    Coroutine *co = qemu_coroutine_create(process_incoming_migration_co);



    migrate_decompress_threads_create();

    qemu_file_set_blocking(f, false);

    qemu_coroutine_enter(co, f);

}
