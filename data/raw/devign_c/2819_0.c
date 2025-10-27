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



    if (!qtest_driver() &&

        fs.f_type != HUGETLBFS_MAGIC) {

        fprintf(stderr, "Warning: path not on HugeTLBFS: %s\n", path);

    }



    return fs.f_bsize;

}
