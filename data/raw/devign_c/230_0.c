static int local_create_mapped_attr_dir(FsContext *ctx, const char *path)

{

    int err;

    char attr_dir[PATH_MAX];

    char *tmp_path = g_strdup(path);



    snprintf(attr_dir, PATH_MAX, "%s/%s/%s",

             ctx->fs_root, dirname(tmp_path), VIRTFS_META_DIR);



    err = mkdir(attr_dir, 0700);

    if (err < 0 && errno == EEXIST) {

        err = 0;

    }

    g_free(tmp_path);

    return err;

}
