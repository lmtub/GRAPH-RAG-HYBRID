static bool local_is_mapped_file_metadata(FsContext *fs_ctx, const char *name)

{

    return !strcmp(name, VIRTFS_META_DIR);

}
