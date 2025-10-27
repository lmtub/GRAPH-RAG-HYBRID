void replay_configure(QemuOpts *opts)

{

    const char *fname;

    const char *rr;

    ReplayMode mode = REPLAY_MODE_NONE;

    Location loc;



    if (!opts) {

        return;

    }



    loc_push_none(&loc);

    qemu_opts_loc_restore(opts);



    rr = qemu_opt_get(opts, "rr");

    if (!rr) {

        /* Just enabling icount */

        return;

    } else if (!strcmp(rr, "record")) {

        mode = REPLAY_MODE_RECORD;

    } else if (!strcmp(rr, "replay")) {

        mode = REPLAY_MODE_PLAY;

    } else {

        error_report("Invalid icount rr option: %s", rr);

        exit(1);

    }



    fname = qemu_opt_get(opts, "rrfile");

    if (!fname) {

        error_report("File name not specified for replay");

        exit(1);

    }



    replay_enable(fname, mode);



    loc_pop(&loc);

}
