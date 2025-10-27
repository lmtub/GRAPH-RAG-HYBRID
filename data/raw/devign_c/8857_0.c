static int qemu_gluster_create(const char *filename,

                               QemuOpts *opts, Error **errp)

{

    BlockdevOptionsGluster *gconf;

    struct glfs *glfs;

    struct glfs_fd *fd;

    int ret = 0;

    int prealloc = 0;

    int64_t total_size = 0;

    char *tmp = NULL;



    gconf = g_new0(BlockdevOptionsGluster, 1);

    gconf->debug = qemu_opt_get_number_del(opts, GLUSTER_OPT_DEBUG,

                                           GLUSTER_DEBUG_DEFAULT);

    if (gconf->debug < 0) {

        gconf->debug = 0;

    } else if (gconf->debug > GLUSTER_DEBUG_MAX) {

        gconf->debug = GLUSTER_DEBUG_MAX;

    }

    gconf->has_debug = true;



    gconf->logfile = qemu_opt_get_del(opts, GLUSTER_OPT_LOGFILE);

    if (!gconf->logfile) {

        gconf->logfile = g_strdup(GLUSTER_LOGFILE_DEFAULT);

    }

    gconf->has_logfile = true;



    glfs = qemu_gluster_init(gconf, filename, NULL, errp);

    if (!glfs) {

        ret = -errno;

        goto out;

    }



    total_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                          BDRV_SECTOR_SIZE);



    tmp = qemu_opt_get_del(opts, BLOCK_OPT_PREALLOC);

    if (!tmp || !strcmp(tmp, "off")) {

        prealloc = 0;

    } else if (!strcmp(tmp, "full") && gluster_supports_zerofill()) {

        prealloc = 1;

    } else {

        error_setg(errp, "Invalid preallocation mode: '%s'"

                         " or GlusterFS doesn't support zerofill API", tmp);

        ret = -EINVAL;

        goto out;

    }



    fd = glfs_creat(glfs, gconf->path,

                    O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, S_IRUSR | S_IWUSR);

    if (!fd) {

        ret = -errno;

    } else {

        if (!glfs_ftruncate(fd, total_size)) {

            if (prealloc && qemu_gluster_zerofill(fd, 0, total_size)) {

                ret = -errno;

            }

        } else {

            ret = -errno;

        }



        if (glfs_close(fd) != 0) {

            ret = -errno;

        }

    }

out:

    g_free(tmp);

    qapi_free_BlockdevOptionsGluster(gconf);

    glfs_clear_preopened(glfs);

    return ret;

}
