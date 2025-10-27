void do_m68k_simcall(CPUM68KState *env, int nr)

{

    uint32_t *args;



    args = (uint32_t *)(env->aregs[7] + 4);

    switch (nr) {

    case SYS_EXIT:

        exit(ARG(0));

    case SYS_READ:

        check_err(env, read(ARG(0), (void *)ARG(1), ARG(2)));

        break;

    case SYS_WRITE:

        check_err(env, write(ARG(0), (void *)ARG(1), ARG(2)));

        break;

    case SYS_OPEN:

        check_err(env, open((char *)ARG(0), translate_openflags(ARG(1)),

                            ARG(2)));

        break;

    case SYS_CLOSE:

        {

            /* Ignore attempts to close stdin/out/err.  */

            int fd = ARG(0);

            if (fd > 2)

              check_err(env, close(fd));

            else

              check_err(env, 0);

            break;

        }

    case SYS_BRK:

        {

            int32_t ret;



            ret = do_brk((void *)ARG(0));

            if (ret == -ENOMEM)

                ret = -1;

            check_err(env, ret);

        }

        break;

    case SYS_FSTAT:

        {

            struct stat s;

            int rc;

            struct m86k_sim_stat *p;

            rc = check_err(env, fstat(ARG(0), &s));

            if (rc == 0) {

                p = (struct m86k_sim_stat *)ARG(1);

                p->sim_st_dev = tswap16(s.st_dev);

                p->sim_st_ino = tswap16(s.st_ino);

                p->sim_st_mode = tswap32(s.st_mode);

                p->sim_st_nlink = tswap16(s.st_nlink);

                p->sim_st_uid = tswap16(s.st_uid);

                p->sim_st_gid = tswap16(s.st_gid);

                p->sim_st_rdev = tswap16(s.st_rdev);

                p->sim_st_size = tswap32(s.st_size);

                p->sim_st_atime = tswap32(s.st_atime);

                p->sim_st_mtime = tswap32(s.st_mtime);

                p->sim_st_ctime = tswap32(s.st_ctime);

                p->sim_st_blksize = tswap32(s.st_blksize);

                p->sim_st_blocks = tswap32(s.st_blocks);

            }

        }

        break;

    case SYS_ISATTY:

        check_err(env, isatty(ARG(0)));

        break;

    case SYS_LSEEK:

        check_err(env, lseek(ARG(0), (int32_t)ARG(1), ARG(2)));

        break;

    default:

        cpu_abort(env, "Unsupported m68k sim syscall %d\n", nr);

    }

}
