static const char *local_mapped_attr_path(FsContext *ctx,

                                          const char *path, char *buffer)

{

    char *dir_name;

    char *tmp_path = strdup(path);

    char *base_name = basename(tmp_path);



    /* NULL terminate the directory */

    dir_name = tmp_path;

    *(base_name - 1) = '\0';



    snprintf(buffer, PATH_MAX, "%s/%s/%s/%s",

             ctx->fs_root, dir_name, VIRTFS_META_DIR, base_name);

    free(tmp_path);

    return buffer;

}
