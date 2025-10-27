static int balloon_parse(const char *arg)

{

    QemuOpts *opts;



    if (strcmp(arg, "none") == 0) {

        return 0;

    }



    if (!strncmp(arg, "virtio", 6)) {

        if (arg[6] == ',') {

            /* have params -> parse them */

            opts = qemu_opts_parse(qemu_find_opts("device"), arg+7, 0);

            if (!opts)

                return  -1;

        } else {

            /* create empty opts */

            opts = qemu_opts_create(qemu_find_opts("device"), NULL, 0);

        }

        qemu_opt_set(opts, "driver", "virtio-balloon");

        return 0;

    }



    return -1;

}
