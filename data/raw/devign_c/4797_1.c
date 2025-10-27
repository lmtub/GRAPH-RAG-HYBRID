static int raw_open_common(BlockDriverState *bs, QDict *options,
                           int bdrv_flags, int open_flags, Error **errp)
{
    BDRVRawState *s = bs->opaque;
    QemuOpts *opts;
    Error *local_err = NULL;
    const char *filename = NULL;
    const char *str;
    BlockdevAioOptions aio, aio_default;
    int fd, ret;
    struct stat st;
    OnOffAuto locking;
    opts = qemu_opts_create(&raw_runtime_opts, NULL, 0, &error_abort);
    qemu_opts_absorb_qdict(opts, options, &local_err);
    filename = qemu_opt_get(opts, "filename");
    ret = raw_normalize_devicepath(&filename);
    if (ret != 0) {
        error_setg_errno(errp, -ret, "Could not normalize device path");
    aio_default = (bdrv_flags & BDRV_O_NATIVE_AIO)
                  ? BLOCKDEV_AIO_OPTIONS_NATIVE
                  : BLOCKDEV_AIO_OPTIONS_THREADS;
    aio = qapi_enum_parse(&BlockdevAioOptions_lookup,
                          qemu_opt_get(opts, "aio"),
                          aio_default, &local_err);
    s->use_linux_aio = (aio == BLOCKDEV_AIO_OPTIONS_NATIVE);
    locking = qapi_enum_parse(&OnOffAuto_lookup,
                              qemu_opt_get(opts, "locking"),
                              ON_OFF_AUTO_AUTO, &local_err);
    switch (locking) {
    case ON_OFF_AUTO_ON:
        s->use_lock = true;
        if (!qemu_has_ofd_lock()) {
            fprintf(stderr,
                    "File lock requested but OFD locking syscall is "
                    "unavailable, falling back to POSIX file locks.\n"
                    "Due to the implementation, locks can be lost "
                    "unexpectedly.\n");
        break;
    case ON_OFF_AUTO_OFF:
        s->use_lock = false;
        break;
    case ON_OFF_AUTO_AUTO:
        s->use_lock = qemu_has_ofd_lock();
        break;
    default:
        abort();
    s->open_flags = open_flags;
    raw_parse_flags(bdrv_flags, &s->open_flags);
    s->fd = -1;
    fd = qemu_open(filename, s->open_flags, 0644);
    if (fd < 0) {
        ret = -errno;
        error_setg_errno(errp, errno, "Could not open '%s'", filename);
        if (ret == -EROFS) {
            ret = -EACCES;
    s->fd = fd;
    s->lock_fd = -1;
    if (s->use_lock) {
        fd = qemu_open(filename, s->open_flags);
        if (fd < 0) {
            ret = -errno;
            error_setg_errno(errp, errno, "Could not open '%s' for locking",
                             filename);
            qemu_close(s->fd);
        s->lock_fd = fd;
    s->perm = 0;
    s->shared_perm = BLK_PERM_ALL;
#ifdef CONFIG_LINUX_AIO
     /* Currently Linux does AIO only for files opened with O_DIRECT */
    if (s->use_linux_aio && !(s->open_flags & O_DIRECT)) {
        error_setg(errp, "aio=native was specified, but it requires "
                         "cache.direct=on, which was not specified.");
#else
    if (s->use_linux_aio) {
        error_setg(errp, "aio=native was specified, but is not supported "
                         "in this build.");
#endif /* !defined(CONFIG_LINUX_AIO) */
    s->has_discard = true;
    s->has_write_zeroes = true;
    bs->supported_zero_flags = BDRV_REQ_MAY_UNMAP;
    if ((bs->open_flags & BDRV_O_NOCACHE) != 0) {
        s->needs_alignment = true;
    if (fstat(s->fd, &st) < 0) {
        ret = -errno;
        error_setg_errno(errp, errno, "Could not stat file");
    if (S_ISREG(st.st_mode)) {
        s->discard_zeroes = true;
        s->has_fallocate = true;
    if (S_ISBLK(st.st_mode)) {
#ifdef BLKDISCARDZEROES
        unsigned int arg;
        if (ioctl(s->fd, BLKDISCARDZEROES, &arg) == 0 && arg) {
            s->discard_zeroes = true;
#endif
#ifdef __linux__
        /* On Linux 3.10, BLKDISCARD leaves stale data in the page cache.  Do
         * not rely on the contents of discarded blocks unless using O_DIRECT.
         * Same for BLKZEROOUT.
         */
        if (!(bs->open_flags & BDRV_O_NOCACHE)) {
            s->discard_zeroes = false;
            s->has_write_zeroes = false;
#endif
#ifdef __FreeBSD__
    if (S_ISCHR(st.st_mode)) {
        /*
         * The file is a char device (disk), which on FreeBSD isn't behind
         * a pager, so force all requests to be aligned. This is needed
         * so QEMU makes sure all IO operations on the device are aligned
         * to sector size, or else FreeBSD will reject them with EINVAL.
         */
        s->needs_alignment = true;
#endif
#ifdef CONFIG_XFS
    if (platform_test_xfs_fd(s->fd)) {
        s->is_xfs = true;
#endif
    ret = 0;
fail:
    if (filename && (bdrv_flags & BDRV_O_TEMPORARY)) {
        unlink(filename);
    qemu_opts_del(opts);
    return ret;