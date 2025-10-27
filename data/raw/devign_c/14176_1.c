static void local_mapped_file_attr(FsContext *ctx, const char *path,

                                   struct stat *stbuf)

{

    FILE *fp;

    char buf[ATTR_MAX];

    char *attr_path;



    attr_path = local_mapped_attr_path(ctx, path);

    fp = local_fopen(attr_path, "r");

    g_free(attr_path);

    if (!fp) {

        return;

    }

    memset(buf, 0, ATTR_MAX);

    while (fgets(buf, ATTR_MAX, fp)) {

        if (!strncmp(buf, "virtfs.uid", 10)) {

            stbuf->st_uid = atoi(buf+11);

        } else if (!strncmp(buf, "virtfs.gid", 10)) {

            stbuf->st_gid = atoi(buf+11);

        } else if (!strncmp(buf, "virtfs.mode", 11)) {

            stbuf->st_mode = atoi(buf+12);

        } else if (!strncmp(buf, "virtfs.rdev", 11)) {

            stbuf->st_rdev = atoi(buf+12);

        }

        memset(buf, 0, ATTR_MAX);

    }

    fclose(fp);

}
