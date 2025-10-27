static off_t proxy_telldir(FsContext *ctx, V9fsFidOpenState *fs)

{

    return telldir(fs->dir);

}
