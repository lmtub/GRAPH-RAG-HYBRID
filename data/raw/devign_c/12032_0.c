static int proxy_statfs(FsContext *s, V9fsPath *fs_path, struct statfs *stbuf)

{

    int retval;

    retval = v9fs_request(s->private, T_STATFS, stbuf, "s", fs_path);

    if (retval < 0) {

        errno = -retval;

        return -1;

    }

    return retval;

}
