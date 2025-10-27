static int qcow2_create(const char *filename, QEMUOptionParameter *options)

{

    const char *backing_file = NULL;

    const char *backing_fmt = NULL;

    uint64_t sectors = 0;

    int flags = 0;

    size_t cluster_size = DEFAULT_CLUSTER_SIZE;

    int prealloc = 0;



    /* Read out options */

    while (options && options->name) {

        if (!strcmp(options->name, BLOCK_OPT_SIZE)) {

            sectors = options->value.n / 512;

        } else if (!strcmp(options->name, BLOCK_OPT_BACKING_FILE)) {

            backing_file = options->value.s;

        } else if (!strcmp(options->name, BLOCK_OPT_BACKING_FMT)) {

            backing_fmt = options->value.s;

        } else if (!strcmp(options->name, BLOCK_OPT_ENCRYPT)) {

            flags |= options->value.n ? BLOCK_FLAG_ENCRYPT : 0;

        } else if (!strcmp(options->name, BLOCK_OPT_CLUSTER_SIZE)) {

            if (options->value.n) {

                cluster_size = options->value.n;

            }

        } else if (!strcmp(options->name, BLOCK_OPT_PREALLOC)) {

            if (!options->value.s || !strcmp(options->value.s, "off")) {

                prealloc = 0;

            } else if (!strcmp(options->value.s, "metadata")) {

                prealloc = 1;

            } else {

                fprintf(stderr, "Invalid preallocation mode: '%s'\n",

                    options->value.s);

                return -EINVAL;

            }

        }

        options++;

    }



    if (backing_file && prealloc) {

        fprintf(stderr, "Backing file and preallocation cannot be used at "

            "the same time\n");

        return -EINVAL;

    }



    return qcow2_create2(filename, sectors, backing_file, backing_fmt, flags,

                         cluster_size, prealloc, options);

}
