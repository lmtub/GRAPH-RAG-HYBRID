void do_m68k_semihosting(CPUM68KState *env, int nr)

{

    uint32_t args;

    void *p;

    void *q;

    uint32_t len;

    uint32_t result;



    args = env->dregs[1];

    switch (nr) {

    case HOSTED_EXIT:

        gdb_exit(env, env->dregs[0]);

        exit(env->dregs[0]);

    case HOSTED_OPEN:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "open,%s,%x,%x", ARG(0), (int)ARG(1),

                           ARG(2), ARG(3));

            return;

        } else {

            if (!(p = lock_user_string(ARG(0)))) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = open(p, translate_openflags(ARG(2)), ARG(3));

                unlock_user(p, ARG(0), 0);

            }

        }

        break;

    case HOSTED_CLOSE:

        {

            /* Ignore attempts to close stdin/out/err.  */

            int fd = ARG(0);

            if (fd > 2) {

                if (use_gdb_syscalls()) {

                    gdb_do_syscall(m68k_semi_cb, "close,%x", ARG(0));

                    return;

                } else {

                    result = close(fd);

                }

            } else {

                result = 0;

            }

            break;

        }

    case HOSTED_READ:

        len = ARG(2);

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "read,%x,%x,%x",

                           ARG(0), ARG(1), len);

            return;

        } else {

            if (!(p = lock_user(VERIFY_WRITE, ARG(1), len, 0))) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = read(ARG(0), p, len);

                unlock_user(p, ARG(1), len);

            }

        }

        break;

    case HOSTED_WRITE:

        len = ARG(2);

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "write,%x,%x,%x",

                           ARG(0), ARG(1), len);

            return;

        } else {

            if (!(p = lock_user(VERIFY_READ, ARG(1), len, 1))) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = write(ARG(0), p, len);

                unlock_user(p, ARG(0), 0);

            }

        }

        break;

    case HOSTED_LSEEK:

        {

            uint64_t off;

            off = (uint32_t)ARG(2) | ((uint64_t)ARG(1) << 32);

            if (use_gdb_syscalls()) {

                m68k_semi_is_fseek = 1;

                gdb_do_syscall(m68k_semi_cb, "fseek,%x,%lx,%x",

                               ARG(0), off, ARG(3));

            } else {

                off = lseek(ARG(0), off, ARG(3));

                /* FIXME - handle put_user() failure */

                put_user_u32(off >> 32, args);

                put_user_u32(off, args + 4);

                put_user_u32(errno, args + 8);

            }

            return;

        }

    case HOSTED_RENAME:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "rename,%s,%s",

                           ARG(0), (int)ARG(1), ARG(2), (int)ARG(3));

            return;

        } else {

            p = lock_user_string(ARG(0));

            q = lock_user_string(ARG(2));

            if (!p || !q) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = rename(p, q);

            }

            unlock_user(p, ARG(0), 0);

            unlock_user(q, ARG(2), 0);

        }

        break;

    case HOSTED_UNLINK:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "unlink,%s",

                           ARG(0), (int)ARG(1));

            return;

        } else {

            if (!(p = lock_user_string(ARG(0)))) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = unlink(p);

                unlock_user(p, ARG(0), 0);

            }

        }

        break;

    case HOSTED_STAT:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "stat,%s,%x",

                           ARG(0), (int)ARG(1), ARG(2));

            return;

        } else {

            struct stat s;

            if (!(p = lock_user_string(ARG(0)))) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = stat(p, &s);

                unlock_user(p, ARG(0), 0);

            }

            if (result == 0) {

                translate_stat(env, ARG(2), &s);

            }

        }

        break;

    case HOSTED_FSTAT:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "fstat,%x,%x",

                           ARG(0), ARG(1));

            return;

        } else {

            struct stat s;

            result = fstat(ARG(0), &s);

            if (result == 0) {

                translate_stat(env, ARG(1), &s);

            }

        }

        break;

    case HOSTED_GETTIMEOFDAY:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "gettimeofday,%x,%x",

                           ARG(0), ARG(1));

            return;

        } else {

            qemu_timeval tv;

            struct gdb_timeval *p;

            result = qemu_gettimeofday(&tv);

            if (result != 0) {

                if (!(p = lock_user(VERIFY_WRITE,

                                    ARG(0), sizeof(struct gdb_timeval), 0))) {

                    /* FIXME - check error code? */

                    result = -1;

                } else {

                    p->tv_sec = cpu_to_be32(tv.tv_sec);

                    p->tv_usec = cpu_to_be64(tv.tv_usec);

                    unlock_user(p, ARG(0), sizeof(struct gdb_timeval));

                }

            }

        }

        break;

    case HOSTED_ISATTY:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "isatty,%x", ARG(0));

            return;

        } else {

            result = isatty(ARG(0));

        }

        break;

    case HOSTED_SYSTEM:

        if (use_gdb_syscalls()) {

            gdb_do_syscall(m68k_semi_cb, "system,%s",

                           ARG(0), (int)ARG(1));

            return;

        } else {

            if (!(p = lock_user_string(ARG(0)))) {

                /* FIXME - check error code? */

                result = -1;

            } else {

                result = system(p);

                unlock_user(p, ARG(0), 0);

            }

        }

        break;

    case HOSTED_INIT_SIM:

#if defined(CONFIG_USER_ONLY)

        {

        TaskState *ts = env->opaque;

        /* Allocate the heap using sbrk.  */

        if (!ts->heap_limit) {

            long ret;

            uint32_t size;

            uint32_t base;



            base = do_brk(0);

            size = SEMIHOSTING_HEAP_SIZE;

            /* Try a big heap, and reduce the size if that fails.  */

            for (;;) {

                ret = do_brk(base + size);

                if (ret != -1)

                    break;

                size >>= 1;

            }

            ts->heap_limit = base + size;

        }

        /* This call may happen before we have writable memory, so return

           values directly in registers.  */

        env->dregs[1] = ts->heap_limit;

        env->aregs[7] = ts->stack_base;

        }

#else

        /* FIXME: This is wrong for boards where RAM does not start at

           address zero.  */

        env->dregs[1] = ram_size;

        env->aregs[7] = ram_size;

#endif

        return;

    default:

        cpu_abort(env, "Unsupported semihosting syscall %d\n", nr);

        result = 0;

    }

    /* FIXME - handle put_user() failure */

    put_user_u32(result, args);

    put_user_u32(errno, args + 4);

}
