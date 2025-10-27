static long gethugepagesize(const char *path, Error **errp)

{

    struct statfs fs;

    int ret;



    do {

        ret = statfs(path, &fs);

    } while (ret != 0 && errno == EINTR);



    if (ret != 0) {

        error_setg_errno(errp, errno, "failed to get page size of file %s",

                         path);

        return 0;

    }



    return fs.f_bsize;

}
