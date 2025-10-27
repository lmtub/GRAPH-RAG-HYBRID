long do_syscall(void *cpu_env, int num, long arg1, long arg2, long arg3, 

                long arg4, long arg5, long arg6)

{

    long ret;

    struct stat st;

    struct kernel_statfs *stfs;

    

#ifdef DEBUG

    gemu_log("syscall %d\n", num);

#endif

    switch(num) {

    case TARGET_NR_exit:

#ifdef HAVE_GPROF

        _mcleanup();

#endif

        _exit(arg1);

        ret = 0; /* avoid warning */

        break;

    case TARGET_NR_read:

        ret = get_errno(read(arg1, (void *)arg2, arg3));

        break;

    case TARGET_NR_write:

        ret = get_errno(write(arg1, (void *)arg2, arg3));

        break;

    case TARGET_NR_open:

        ret = get_errno(open((const char *)arg1, arg2, arg3));

        break;

    case TARGET_NR_close:

        ret = get_errno(close(arg1));

        break;

    case TARGET_NR_brk:

        ret = do_brk((char *)arg1);

        break;

    case TARGET_NR_fork:

        ret = get_errno(fork());

        break;

    case TARGET_NR_waitpid:

        {

            int *status = (int *)arg2;

            ret = get_errno(waitpid(arg1, status, arg3));

            if (!is_error(ret) && status)

                tswapls((long *)&status);

        }

        break;

    case TARGET_NR_creat:

        ret = get_errno(creat((const char *)arg1, arg2));

        break;

    case TARGET_NR_link:

        ret = get_errno(link((const char *)arg1, (const char *)arg2));

        break;

    case TARGET_NR_unlink:

        ret = get_errno(unlink((const char *)arg1));

        break;

    case TARGET_NR_execve:

        ret = get_errno(execve((const char *)arg1, (void *)arg2, (void *)arg3));

        break;

    case TARGET_NR_chdir:

        ret = get_errno(chdir((const char *)arg1));

        break;

    case TARGET_NR_time:

        {

            int *time_ptr = (int *)arg1;

            ret = get_errno(time((time_t *)time_ptr));

            if (!is_error(ret) && time_ptr)

                tswap32s(time_ptr);

        }

        break;

    case TARGET_NR_mknod:

        ret = get_errno(mknod((const char *)arg1, arg2, arg3));

        break;

    case TARGET_NR_chmod:

        ret = get_errno(chmod((const char *)arg1, arg2));

        break;

    case TARGET_NR_lchown:

        ret = get_errno(chown((const char *)arg1, arg2, arg3));

        break;

    case TARGET_NR_break:

        goto unimplemented;

    case TARGET_NR_oldstat:

        goto unimplemented;

    case TARGET_NR_lseek:

        ret = get_errno(lseek(arg1, arg2, arg3));

        break;

    case TARGET_NR_getpid:

        ret = get_errno(getpid());

        break;

    case TARGET_NR_mount:

        /* need to look at the data field */

        goto unimplemented;

    case TARGET_NR_umount:

        ret = get_errno(umount((const char *)arg1));

        break;

    case TARGET_NR_setuid:

        ret = get_errno(setuid(arg1));

        break;

    case TARGET_NR_getuid:

        ret = get_errno(getuid());

        break;

    case TARGET_NR_stime:

        {

            int *time_ptr = (int *)arg1;

            if (time_ptr)

                tswap32s(time_ptr);

            ret = get_errno(stime((time_t *)time_ptr));

        }

        break;

    case TARGET_NR_ptrace:

        goto unimplemented;

    case TARGET_NR_alarm:

        ret = alarm(arg1);

        break;

    case TARGET_NR_oldfstat:

        goto unimplemented;

    case TARGET_NR_pause:

        ret = get_errno(pause());

        break;

    case TARGET_NR_utime:

        goto unimplemented;

    case TARGET_NR_stty:

        goto unimplemented;

    case TARGET_NR_gtty:

        goto unimplemented;

    case TARGET_NR_access:

        ret = get_errno(access((const char *)arg1, arg2));

        break;

    case TARGET_NR_nice:

        ret = get_errno(nice(arg1));

        break;

    case TARGET_NR_ftime:

        goto unimplemented;

    case TARGET_NR_sync:

        ret = get_errno(sync());

        break;

    case TARGET_NR_kill:

        ret = get_errno(kill(arg1, arg2));

        break;

    case TARGET_NR_rename:

        ret = get_errno(rename((const char *)arg1, (const char *)arg2));

        break;

    case TARGET_NR_mkdir:

        ret = get_errno(mkdir((const char *)arg1, arg2));

        break;

    case TARGET_NR_rmdir:

        ret = get_errno(rmdir((const char *)arg1));

        break;

    case TARGET_NR_dup:

        ret = get_errno(dup(arg1));

        break;

    case TARGET_NR_pipe:

        {

            int *pipe_ptr = (int *)arg1;

            ret = get_errno(pipe(pipe_ptr));

            if (!is_error(ret)) {

                tswap32s(&pipe_ptr[0]);

                tswap32s(&pipe_ptr[1]);

            }

        }

        break;

    case TARGET_NR_times:

        goto unimplemented;

    case TARGET_NR_prof:

        goto unimplemented;

    case TARGET_NR_setgid:

        ret = get_errno(setgid(arg1));

        break;

    case TARGET_NR_getgid:

        ret = get_errno(getgid());

        break;

    case TARGET_NR_signal:

        goto unimplemented;

    case TARGET_NR_geteuid:

        ret = get_errno(geteuid());

        break;

    case TARGET_NR_getegid:

        ret = get_errno(getegid());

        break;

    case TARGET_NR_acct:

        goto unimplemented;

    case TARGET_NR_umount2:

        ret = get_errno(umount2((const char *)arg1, arg2));

        break;

    case TARGET_NR_lock:

        goto unimplemented;

    case TARGET_NR_ioctl:

        ret = do_ioctl(arg1, arg2, arg3);

        break;

    case TARGET_NR_fcntl:

        switch(arg2) {

        case F_GETLK:

        case F_SETLK:

        case F_SETLKW:

            goto unimplemented;

        default:

            ret = get_errno(fcntl(arg1, arg2, arg3));

            break;

        }

        break;

    case TARGET_NR_mpx:

        goto unimplemented;

    case TARGET_NR_setpgid:

        ret = get_errno(setpgid(arg1, arg2));

        break;

    case TARGET_NR_ulimit:

        goto unimplemented;

    case TARGET_NR_oldolduname:

        goto unimplemented;

    case TARGET_NR_umask:

        ret = get_errno(umask(arg1));

        break;

    case TARGET_NR_chroot:

        ret = get_errno(chroot((const char *)arg1));

        break;

    case TARGET_NR_ustat:

        goto unimplemented;

    case TARGET_NR_dup2:

        ret = get_errno(dup2(arg1, arg2));

        break;

    case TARGET_NR_getppid:

        ret = get_errno(getppid());

        break;

    case TARGET_NR_getpgrp:

        ret = get_errno(getpgrp());

        break;

    case TARGET_NR_setsid:

        ret = get_errno(setsid());

        break;

    case TARGET_NR_sigaction:

#if 0

        {

            int signum = arg1;

            struct target_old_sigaction *tact = arg2, *toldact = arg3;

            ret = get_errno(setsid());

            



        }

        break;

#else

        goto unimplemented;

#endif

    case TARGET_NR_sgetmask:

        goto unimplemented;

    case TARGET_NR_ssetmask:

        goto unimplemented;

    case TARGET_NR_setreuid:

        ret = get_errno(setreuid(arg1, arg2));

        break;

    case TARGET_NR_setregid:

        ret = get_errno(setregid(arg1, arg2));

        break;

    case TARGET_NR_sigsuspend:

        goto unimplemented;

    case TARGET_NR_sigpending:

        goto unimplemented;

    case TARGET_NR_sethostname:

        ret = get_errno(sethostname((const char *)arg1, arg2));

        break;

    case TARGET_NR_setrlimit:

        goto unimplemented;

    case TARGET_NR_getrlimit:

        goto unimplemented;

    case TARGET_NR_getrusage:

        goto unimplemented;

    case TARGET_NR_gettimeofday:

        {

            struct target_timeval *target_tv = (void *)arg1;

            struct timeval tv;

            ret = get_errno(gettimeofday(&tv, NULL));

            if (!is_error(ret)) {

                target_tv->tv_sec = tswapl(tv.tv_sec);

                target_tv->tv_usec = tswapl(tv.tv_usec);

            }

        }

        break;

    case TARGET_NR_settimeofday:

        {

            struct target_timeval *target_tv = (void *)arg1;

            struct timeval tv;

            tv.tv_sec = tswapl(target_tv->tv_sec);

            tv.tv_usec = tswapl(target_tv->tv_usec);

            ret = get_errno(settimeofday(&tv, NULL));

        }

        break;

    case TARGET_NR_getgroups:

        goto unimplemented;

    case TARGET_NR_setgroups:

        goto unimplemented;

    case TARGET_NR_select:

        goto unimplemented;

    case TARGET_NR_symlink:

        ret = get_errno(symlink((const char *)arg1, (const char *)arg2));

        break;

    case TARGET_NR_oldlstat:

        goto unimplemented;

    case TARGET_NR_readlink:

        ret = get_errno(readlink((const char *)arg1, (char *)arg2, arg3));

        break;

    case TARGET_NR_uselib:

        goto unimplemented;

    case TARGET_NR_swapon:

        ret = get_errno(swapon((const char *)arg1, arg2));

        break;

    case TARGET_NR_reboot:

        goto unimplemented;

    case TARGET_NR_readdir:

        goto unimplemented;

#ifdef TARGET_I386

    case TARGET_NR_mmap:

        {

            uint32_t v1, v2, v3, v4, v5, v6, *vptr;

            vptr = (uint32_t *)arg1;

            v1 = tswap32(vptr[0]);

            v2 = tswap32(vptr[1]);

            v3 = tswap32(vptr[2]);

            v4 = tswap32(vptr[3]);

            v5 = tswap32(vptr[4]);

            v6 = tswap32(vptr[5]);

            ret = get_errno((long)mmap((void *)v1, v2, v3, v4, v5, v6));

        }

        break;

#endif

#ifdef TARGET_I386

    case TARGET_NR_mmap2:

#else

    case TARGET_NR_mmap:

#endif

        ret = get_errno((long)mmap((void *)arg1, arg2, arg3, arg4, arg5, arg6));

        break;

    case TARGET_NR_munmap:

        ret = get_errno(munmap((void *)arg1, arg2));

        break;

    case TARGET_NR_truncate:

        ret = get_errno(truncate((const char *)arg1, arg2));

        break;

    case TARGET_NR_ftruncate:

        ret = get_errno(ftruncate(arg1, arg2));

        break;

    case TARGET_NR_fchmod:

        ret = get_errno(fchmod(arg1, arg2));

        break;

    case TARGET_NR_fchown:

        ret = get_errno(fchown(arg1, arg2, arg3));

        break;

    case TARGET_NR_getpriority:

        ret = get_errno(getpriority(arg1, arg2));

        break;

    case TARGET_NR_setpriority:

        ret = get_errno(setpriority(arg1, arg2, arg3));

        break;

    case TARGET_NR_profil:

        goto unimplemented;

    case TARGET_NR_statfs:

        stfs = (void *)arg2;

        ret = get_errno(sys_statfs((const char *)arg1, stfs));

    convert_statfs:

        if (!is_error(ret)) {

            tswap32s(&stfs->f_type);

            tswap32s(&stfs->f_bsize);

            tswap32s(&stfs->f_blocks);

            tswap32s(&stfs->f_bfree);

            tswap32s(&stfs->f_bavail);

            tswap32s(&stfs->f_files);

            tswap32s(&stfs->f_ffree);

            tswap32s(&stfs->f_fsid.val[0]);

            tswap32s(&stfs->f_fsid.val[1]);

            tswap32s(&stfs->f_namelen);

        }

        break;

    case TARGET_NR_fstatfs:

        stfs = (void *)arg2;

        ret = get_errno(sys_fstatfs(arg1, stfs));

        goto convert_statfs;

    case TARGET_NR_ioperm:

        goto unimplemented;

    case TARGET_NR_socketcall:

        ret = do_socketcall(arg1, (long *)arg2);

        break;

    case TARGET_NR_syslog:

        goto unimplemented;

    case TARGET_NR_setitimer:

        goto unimplemented;

    case TARGET_NR_getitimer:

        goto unimplemented;

    case TARGET_NR_stat:

        ret = get_errno(stat((const char *)arg1, &st));

        goto do_stat;

    case TARGET_NR_lstat:

        ret = get_errno(lstat((const char *)arg1, &st));

        goto do_stat;

    case TARGET_NR_fstat:

        {

            ret = get_errno(fstat(arg1, &st));

        do_stat:

            if (!is_error(ret)) {

                struct target_stat *target_st = (void *)arg2;

                target_st->st_dev = tswap16(st.st_dev);

                target_st->st_ino = tswapl(st.st_ino);

                target_st->st_mode = tswap16(st.st_mode);

                target_st->st_nlink = tswap16(st.st_nlink);

                target_st->st_uid = tswap16(st.st_uid);

                target_st->st_gid = tswap16(st.st_gid);

                target_st->st_rdev = tswap16(st.st_rdev);

                target_st->st_size = tswapl(st.st_size);

                target_st->st_blksize = tswapl(st.st_blksize);

                target_st->st_blocks = tswapl(st.st_blocks);

                target_st->st_atime = tswapl(st.st_atime);

                target_st->st_mtime = tswapl(st.st_mtime);

                target_st->st_ctime = tswapl(st.st_ctime);

            }

        }

        break;

    case TARGET_NR_olduname:

        goto unimplemented;

    case TARGET_NR_iopl:

        goto unimplemented;

    case TARGET_NR_vhangup:

        ret = get_errno(vhangup());

        break;

    case TARGET_NR_idle:

        goto unimplemented;

    case TARGET_NR_vm86old:

        goto unimplemented;

    case TARGET_NR_wait4:

        {

            int status;

            target_long *status_ptr = (void *)arg2;

            struct rusage rusage, *rusage_ptr;

            struct target_rusage *target_rusage = (void *)arg4;

            if (target_rusage)

                rusage_ptr = &rusage;

            else

                rusage_ptr = NULL;

            ret = get_errno(wait4(arg1, &status, arg3, rusage_ptr));

            if (!is_error(ret)) {

                if (status_ptr)

                    *status_ptr = tswap32(status);

                if (target_rusage) {

                    target_rusage->ru_utime.tv_sec = tswapl(rusage.ru_utime.tv_sec);

                    target_rusage->ru_utime.tv_usec = tswapl(rusage.ru_utime.tv_usec);

                    target_rusage->ru_stime.tv_sec = tswapl(rusage.ru_stime.tv_sec);

                    target_rusage->ru_stime.tv_usec = tswapl(rusage.ru_stime.tv_usec);

                    target_rusage->ru_maxrss = tswapl(rusage.ru_maxrss);

                    target_rusage->ru_ixrss = tswapl(rusage.ru_ixrss);

                    target_rusage->ru_idrss = tswapl(rusage.ru_idrss);

                    target_rusage->ru_isrss = tswapl(rusage.ru_isrss);

                    target_rusage->ru_minflt = tswapl(rusage.ru_minflt);

                    target_rusage->ru_majflt = tswapl(rusage.ru_majflt);

                    target_rusage->ru_nswap = tswapl(rusage.ru_nswap);

                    target_rusage->ru_inblock = tswapl(rusage.ru_inblock);

                    target_rusage->ru_oublock = tswapl(rusage.ru_oublock);

                    target_rusage->ru_msgsnd = tswapl(rusage.ru_msgsnd);

                    target_rusage->ru_msgrcv = tswapl(rusage.ru_msgrcv);

                    target_rusage->ru_nsignals = tswapl(rusage.ru_nsignals);

                    target_rusage->ru_nvcsw = tswapl(rusage.ru_nvcsw);

                    target_rusage->ru_nivcsw = tswapl(rusage.ru_nivcsw);

                }

            }

        }

        break;

    case TARGET_NR_swapoff:

        ret = get_errno(swapoff((const char *)arg1));

        break;

    case TARGET_NR_sysinfo:

        goto unimplemented;

    case TARGET_NR_ipc:

        goto unimplemented;

    case TARGET_NR_fsync:

        ret = get_errno(fsync(arg1));

        break;

    case TARGET_NR_sigreturn:

        goto unimplemented;

    case TARGET_NR_clone:

        goto unimplemented;

    case TARGET_NR_setdomainname:

        ret = get_errno(setdomainname((const char *)arg1, arg2));

        break;

    case TARGET_NR_uname:

        /* no need to transcode because we use the linux syscall */

        ret = get_errno(sys_uname((struct new_utsname *)arg1));

        break;

#ifdef TARGET_I386

    case TARGET_NR_modify_ldt:

        ret = get_errno(gemu_modify_ldt(cpu_env, arg1, (void *)arg2, arg3));

        break;

#endif

    case TARGET_NR_adjtimex:

        goto unimplemented;

    case TARGET_NR_mprotect:

        ret = get_errno(mprotect((void *)arg1, arg2, arg3));

        break;

    case TARGET_NR_sigprocmask:

        {

            int how = arg1;

            sigset_t set, oldset, *set_ptr;

            target_ulong *pset = (void *)arg2, *poldset = (void *)arg3;

            

            switch(how) {

            case TARGET_SIG_BLOCK:

                how = SIG_BLOCK;

                break;

            case TARGET_SIG_UNBLOCK:

                how = SIG_UNBLOCK;

                break;

            case TARGET_SIG_SETMASK:

                how = SIG_SETMASK;

                break;

            default:

                ret = -EINVAL;

                goto fail;

            }

            

            if (pset) {

                target_to_host_old_sigset(&set, pset);

                set_ptr = &set;

            } else {

                set_ptr = NULL;

            }

            ret = get_errno(sigprocmask(arg1, set_ptr, &oldset));

            if (!is_error(ret) && poldset) {

                host_to_target_old_sigset(poldset, &oldset);

            }

        }

        break;

    case TARGET_NR_create_module:

    case TARGET_NR_init_module:

    case TARGET_NR_delete_module:

    case TARGET_NR_get_kernel_syms:

        goto unimplemented;

    case TARGET_NR_quotactl:

        goto unimplemented;

    case TARGET_NR_getpgid:

        ret = get_errno(getpgid(arg1));

        break;

    case TARGET_NR_fchdir:

        ret = get_errno(fchdir(arg1));

        break;

    case TARGET_NR_bdflush:

        goto unimplemented;

    case TARGET_NR_sysfs:

        goto unimplemented;

    case TARGET_NR_personality:

        ret = get_errno(mprotect((void *)arg1, arg2, arg3));

        break;

    case TARGET_NR_afs_syscall:

        goto unimplemented;

    case TARGET_NR_setfsuid:

        goto unimplemented;

    case TARGET_NR_setfsgid:

        goto unimplemented;

    case TARGET_NR__llseek:

        {

            int64_t res;

            ret = get_errno(_llseek(arg1, arg2, arg3, &res, arg5));

            *(int64_t *)arg4 = tswap64(res);

        }

        break;

    case TARGET_NR_getdents:

#if TARGET_LONG_SIZE != 4

#error not supported

#endif

        {

            struct dirent *dirp = (void *)arg2;

            long count = arg3;

            ret = get_errno(sys_getdents(arg1, dirp, count));

            if (!is_error(ret)) {

                struct dirent *de;

                int len = ret;

                int reclen;

                de = dirp;

                while (len > 0) {

                    reclen = tswap16(de->d_reclen);

                    if (reclen > len)

                        break;

                    de->d_reclen = reclen;

                    tswapls(&de->d_ino);

                    tswapls(&de->d_off);

                    de = (struct dirent *)((char *)de + reclen);

                    len -= reclen;

                }

            }

        }

        break;

    case TARGET_NR__newselect:

        ret = do_select(arg1, (void *)arg2, (void *)arg3, (void *)arg4, 

                        (void *)arg5);

        break;

    case TARGET_NR_flock:

        goto unimplemented;

    case TARGET_NR_msync:

        ret = get_errno(msync((void *)arg1, arg2, arg3));

        break;

    case TARGET_NR_readv:

        {

            int count = arg3;

            int i;

            struct iovec *vec;

            struct target_iovec *target_vec = (void *)arg2;



            vec = alloca(count * sizeof(struct iovec));

            for(i = 0;i < count; i++) {

                vec[i].iov_base = (void *)tswapl(target_vec[i].iov_base);

                vec[i].iov_len = tswapl(target_vec[i].iov_len);

            }

            ret = get_errno(readv(arg1, vec, count));

        }

        break;

    case TARGET_NR_writev:

        {

            int count = arg3;

            int i;

            struct iovec *vec;

            struct target_iovec *target_vec = (void *)arg2;



            vec = alloca(count * sizeof(struct iovec));

            for(i = 0;i < count; i++) {

                vec[i].iov_base = (void *)tswapl(target_vec[i].iov_base);

                vec[i].iov_len = tswapl(target_vec[i].iov_len);

            }

            ret = get_errno(writev(arg1, vec, count));

        }

        break;

    case TARGET_NR_getsid:

        ret = get_errno(getsid(arg1));

        break;

    case TARGET_NR_fdatasync:

        goto unimplemented;

    case TARGET_NR__sysctl:

        goto unimplemented;

    case TARGET_NR_mlock:

        ret = get_errno(mlock((void *)arg1, arg2));

        break;

    case TARGET_NR_munlock:

        ret = get_errno(munlock((void *)arg1, arg2));

        break;

    case TARGET_NR_mlockall:

        ret = get_errno(mlockall(arg1));

        break;

    case TARGET_NR_munlockall:

        ret = get_errno(munlockall());

        break;

    case TARGET_NR_sched_setparam:

        goto unimplemented;

    case TARGET_NR_sched_getparam:

        goto unimplemented;

    case TARGET_NR_sched_setscheduler:

        goto unimplemented;

    case TARGET_NR_sched_getscheduler:

        goto unimplemented;

    case TARGET_NR_sched_yield:

        ret = get_errno(sched_yield());

        break;

    case TARGET_NR_sched_get_priority_max:

    case TARGET_NR_sched_get_priority_min:

    case TARGET_NR_sched_rr_get_interval:

    case TARGET_NR_nanosleep:

    case TARGET_NR_mremap:

    case TARGET_NR_setresuid:

    case TARGET_NR_getresuid:

    case TARGET_NR_vm86:

    case TARGET_NR_query_module:

    case TARGET_NR_poll:

    case TARGET_NR_nfsservctl:

    case TARGET_NR_setresgid:

    case TARGET_NR_getresgid:

    case TARGET_NR_prctl:

    case TARGET_NR_rt_sigreturn:

    case TARGET_NR_rt_sigaction:

    case TARGET_NR_rt_sigprocmask:

    case TARGET_NR_rt_sigpending:

    case TARGET_NR_rt_sigtimedwait:

    case TARGET_NR_rt_sigqueueinfo:

    case TARGET_NR_rt_sigsuspend:

    case TARGET_NR_pread:

    case TARGET_NR_pwrite:

        goto unimplemented;

    case TARGET_NR_chown:

        ret = get_errno(chown((const char *)arg1, arg2, arg3));

        break;

    case TARGET_NR_getcwd:

        ret = get_errno(sys_getcwd1((char *)arg1, arg2));

        break;

    case TARGET_NR_capget:

    case TARGET_NR_capset:

    case TARGET_NR_sigaltstack:

    case TARGET_NR_sendfile:

    case TARGET_NR_getpmsg:

    case TARGET_NR_putpmsg:

    case TARGET_NR_vfork:

        ret = get_errno(vfork());

        break;

    case TARGET_NR_ugetrlimit:

    case TARGET_NR_truncate64:

    case TARGET_NR_ftruncate64:

    case TARGET_NR_stat64:

    case TARGET_NR_lstat64:

    case TARGET_NR_fstat64:

    case TARGET_NR_lchown32:

    case TARGET_NR_getuid32:

    case TARGET_NR_getgid32:

    case TARGET_NR_geteuid32:

    case TARGET_NR_getegid32:

    case TARGET_NR_setreuid32:

    case TARGET_NR_setregid32:

    case TARGET_NR_getgroups32:

    case TARGET_NR_setgroups32:

    case TARGET_NR_fchown32:

    case TARGET_NR_setresuid32:

    case TARGET_NR_getresuid32:

    case TARGET_NR_setresgid32:

    case TARGET_NR_getresgid32:

    case TARGET_NR_chown32:

    case TARGET_NR_setuid32:

    case TARGET_NR_setgid32:

    case TARGET_NR_setfsuid32:

    case TARGET_NR_setfsgid32:

    case TARGET_NR_pivot_root:

    case TARGET_NR_mincore:

    case TARGET_NR_madvise:

    case TARGET_NR_getdents64:

    case TARGET_NR_fcntl64:

    case TARGET_NR_security:

        goto unimplemented;

    case TARGET_NR_gettid:

        ret = get_errno(gettid());

        break;

    case TARGET_NR_readahead:

    case TARGET_NR_setxattr:

    case TARGET_NR_lsetxattr:

    case TARGET_NR_fsetxattr:

    case TARGET_NR_getxattr:

    case TARGET_NR_lgetxattr:

    case TARGET_NR_fgetxattr:

    case TARGET_NR_listxattr:

    case TARGET_NR_llistxattr:

    case TARGET_NR_flistxattr:

    case TARGET_NR_removexattr:

    case TARGET_NR_lremovexattr:

    case TARGET_NR_fremovexattr:

        goto unimplemented;

    default:

    unimplemented:

        gemu_log("Unsupported syscall: %d\n", num);

        ret = -ENOSYS;

        break;

    }

 fail:

    return ret;

}
