void qmp_drive_backup(DriveBackup *arg, Error **errp)

{

    return do_drive_backup(arg, NULL, errp);

}
