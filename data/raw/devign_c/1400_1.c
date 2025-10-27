int qemu_fsdev_add(QemuOpts *opts)

{

    int i;

    struct FsDriverListEntry *fsle;

    const char *fsdev_id = qemu_opts_id(opts);

    const char *fsdriver = qemu_opt_get(opts, "fsdriver");

    const char *writeout = qemu_opt_get(opts, "writeout");

    bool ro = qemu_opt_get_bool(opts, "readonly", 0);



    if (!fsdev_id) {

        fprintf(stderr, "fsdev: No id specified\n");

        return -1;

    }



    if (fsdriver) {

        for (i = 0; i < ARRAY_SIZE(FsDrivers); i++) {

            if (strcmp(FsDrivers[i].name, fsdriver) == 0) {

                break;

            }

        }



        if (i == ARRAY_SIZE(FsDrivers)) {

            fprintf(stderr, "fsdev: fsdriver %s not found\n", fsdriver);

            return -1;

        }

    } else {

        fprintf(stderr, "fsdev: No fsdriver specified\n");

        return -1;

    }



    fsle = g_malloc0(sizeof(*fsle));

    fsle->fse.fsdev_id = g_strdup(fsdev_id);

    fsle->fse.ops = FsDrivers[i].ops;

    if (writeout) {

        if (!strcmp(writeout, "immediate")) {

            fsle->fse.export_flags |= V9FS_IMMEDIATE_WRITEOUT;

        }

    }

    if (ro) {

        fsle->fse.export_flags |= V9FS_RDONLY;

    } else {

        fsle->fse.export_flags &= ~V9FS_RDONLY;

    }



    if (fsle->fse.ops->parse_opts) {

        if (fsle->fse.ops->parse_opts(opts, &fsle->fse)) {



            return -1;

        }

    }



    QTAILQ_INSERT_TAIL(&fsdriver_entries, fsle, next);

    return 0;

}