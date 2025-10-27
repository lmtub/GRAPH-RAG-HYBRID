static void default_drive(int enable, int snapshot, int use_scsi,

                          BlockInterfaceType type, int index,

                          const char *optstr)

{

    QemuOpts *opts;



    if (type == IF_DEFAULT) {

        type = use_scsi ? IF_SCSI : IF_IDE;

    }



    if (!enable || drive_get_by_index(type, index)) {

        return;

    }



    opts = drive_add(type, index, NULL, optstr);

    if (snapshot) {

        drive_enable_snapshot(opts, NULL);

    }

    if (!drive_init(opts, use_scsi)) {

        exit(1);

    }

}
