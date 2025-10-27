DriveInfo *add_init_drive(const char *optstr)

{

    DriveInfo *dinfo;

    QemuOpts *opts;

    MachineClass *mc;



    opts = drive_def(optstr);

    if (!opts)

        return NULL;



    mc = MACHINE_GET_CLASS(current_machine);

    dinfo = drive_new(opts, mc->block_default_type);

    if (!dinfo) {

        qemu_opts_del(opts);

        return NULL;

    }



    return dinfo;

}
