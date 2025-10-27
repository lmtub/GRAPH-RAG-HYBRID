static inline abi_long host_to_target_stat64(void *cpu_env,

                                             abi_ulong target_addr,

                                             struct stat *host_st)

{

#ifdef TARGET_ARM

    if (((CPUARMState *)cpu_env)->eabi) {

        struct target_eabi_stat64 *target_st;



        if (!lock_user_struct(VERIFY_WRITE, target_st, target_addr, 0))

            return -TARGET_EFAULT;

        memset(target_st, 0, sizeof(struct target_eabi_stat64));

        __put_user(host_st->st_dev, &target_st->st_dev);

        __put_user(host_st->st_ino, &target_st->st_ino);

#ifdef TARGET_STAT64_HAS_BROKEN_ST_INO

        __put_user(host_st->st_ino, &target_st->__st_ino);

#endif

        __put_user(host_st->st_mode, &target_st->st_mode);

        __put_user(host_st->st_nlink, &target_st->st_nlink);

        __put_user(host_st->st_uid, &target_st->st_uid);

        __put_user(host_st->st_gid, &target_st->st_gid);

        __put_user(host_st->st_rdev, &target_st->st_rdev);

        __put_user(host_st->st_size, &target_st->st_size);

        __put_user(host_st->st_blksize, &target_st->st_blksize);

        __put_user(host_st->st_blocks, &target_st->st_blocks);

        __put_user(host_st->st_atime, &target_st->target_st_atime);

        __put_user(host_st->st_mtime, &target_st->target_st_mtime);

        __put_user(host_st->st_ctime, &target_st->target_st_ctime);

        unlock_user_struct(target_st, target_addr, 1);

    } else

#endif

    {

#if TARGET_LONG_BITS == 64

        struct target_stat *target_st;

#else

        struct target_stat64 *target_st;

#endif



        if (!lock_user_struct(VERIFY_WRITE, target_st, target_addr, 0))

            return -TARGET_EFAULT;

        memset(target_st, 0, sizeof(*target_st));

        __put_user(host_st->st_dev, &target_st->st_dev);

        __put_user(host_st->st_ino, &target_st->st_ino);

#ifdef TARGET_STAT64_HAS_BROKEN_ST_INO

        __put_user(host_st->st_ino, &target_st->__st_ino);

#endif

        __put_user(host_st->st_mode, &target_st->st_mode);

        __put_user(host_st->st_nlink, &target_st->st_nlink);

        __put_user(host_st->st_uid, &target_st->st_uid);

        __put_user(host_st->st_gid, &target_st->st_gid);

        __put_user(host_st->st_rdev, &target_st->st_rdev);

        /* XXX: better use of kernel struct */

        __put_user(host_st->st_size, &target_st->st_size);

        __put_user(host_st->st_blksize, &target_st->st_blksize);

        __put_user(host_st->st_blocks, &target_st->st_blocks);

        __put_user(host_st->st_atime, &target_st->target_st_atime);

        __put_user(host_st->st_mtime, &target_st->target_st_mtime);

        __put_user(host_st->st_ctime, &target_st->target_st_ctime);

        unlock_user_struct(target_st, target_addr, 1);

    }



    return 0;

}
