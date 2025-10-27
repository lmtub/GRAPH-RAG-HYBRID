size_t qemu_fd_getpagesize(int fd)

{

#ifdef CONFIG_LINUX

    struct statfs fs;

    int ret;



    if (fd != -1) {

        do {

            ret = fstatfs(fd, &fs);

        } while (ret != 0 && errno == EINTR);



        if (ret == 0 && fs.f_type == HUGETLBFS_MAGIC) {

            return fs.f_bsize;

        }

    }








    return getpagesize();

}