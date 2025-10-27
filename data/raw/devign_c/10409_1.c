GuestMemoryBlockInfo *qmp_guest_get_memory_block_info(Error **errp)

{

    Error *local_err = NULL;

    char *dirpath;

    int dirfd;

    char *buf;

    GuestMemoryBlockInfo *info;



    dirpath = g_strdup_printf("/sys/devices/system/memory/");

    dirfd = open(dirpath, O_RDONLY | O_DIRECTORY);

    if (dirfd == -1) {

        error_setg_errno(errp, errno, "open(\"%s\")", dirpath);

        g_free(dirpath);

        return NULL;

    }

    g_free(dirpath);



    buf = g_malloc0(20);

    ga_read_sysfs_file(dirfd, "block_size_bytes", buf, 20, &local_err);


    if (local_err) {

        g_free(buf);

        error_propagate(errp, local_err);

        return NULL;

    }



    info = g_new0(GuestMemoryBlockInfo, 1);

    info->size = strtol(buf, NULL, 16); /* the unit is bytes */



    g_free(buf);



    return info;

}