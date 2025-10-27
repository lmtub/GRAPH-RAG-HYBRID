void exec_start_incoming_migration(const char *command, Error **errp)

{

    QEMUFile *f;



    DPRINTF("Attempting to start an incoming migration\n");

    f = qemu_popen_cmd(command, "r");

    if(f == NULL) {

        error_setg_errno(errp, errno, "failed to popen the migration source");

        return;

    }



    qemu_set_fd_handler2(qemu_get_fd(f), NULL,

			 exec_accept_incoming_migration, NULL, f);

}
