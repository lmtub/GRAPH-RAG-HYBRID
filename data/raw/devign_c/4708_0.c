static int proxy_fstat(FsContext *fs_ctx, int fid_type,

                       V9fsFidOpenState *fs, struct stat *stbuf)

{

    int fd;



    if (fid_type == P9_FID_DIR) {

        fd = dirfd(fs->dir);

    } else {

        fd = fs->fd;

    }

    return fstat(fd, stbuf);

}
