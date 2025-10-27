void qmp_migrate_cancel(Error **errp)

{

    migrate_fd_cancel(migrate_get_current());

}
