static int drive_init_func(QemuOpts *opts, void *opaque)

{

    int *use_scsi = opaque;



    return drive_init(opts, *use_scsi) == NULL;

}
