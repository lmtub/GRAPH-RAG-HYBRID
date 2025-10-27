void helper_do_semihosting(CPUMIPSState *env)

{

    target_ulong *gpr = env->active_tc.gpr;

    const UHIOp op = gpr[25];

    char *p, *p2;



    switch (op) {

    case UHI_exit:

        qemu_log("UHI(%d): exit(%d)\n", op, (int)gpr[4]);

        exit(gpr[4]);

    case UHI_open:

        GET_TARGET_STRING(p, gpr[4]);

        if (!strcmp("/dev/stdin", p)) {

            gpr[2] = 0;

        } else if (!strcmp("/dev/stdout", p)) {

            gpr[2] = 1;

        } else if (!strcmp("/dev/stderr", p)) {

            gpr[2] = 2;

        } else {

            gpr[2] = open(p, get_open_flags(gpr[5]), gpr[6]);

            gpr[3] = errno_mips(errno);

        }

        FREE_TARGET_STRING(p, gpr[4]);

        break;

    case UHI_close:

        if (gpr[4] < 3) {

            /* ignore closing stdin/stdout/stderr */

            gpr[2] = 0;

            goto uhi_done;

        }

        gpr[2] = close(gpr[4]);

        gpr[3] = errno_mips(errno);

        break;

    case UHI_read:

        gpr[2] = read_from_file(env, gpr[4], gpr[5], gpr[6], 0);

        gpr[3] = errno_mips(errno);

        break;

    case UHI_write:

        gpr[2] = write_to_file(env, gpr[4], gpr[5], gpr[6], 0);

        gpr[3] = errno_mips(errno);

        break;

    case UHI_lseek:

        gpr[2] = lseek(gpr[4], gpr[5], gpr[6]);

        gpr[3] = errno_mips(errno);

        break;

    case UHI_unlink:

        GET_TARGET_STRING(p, gpr[4]);

        gpr[2] = remove(p);

        gpr[3] = errno_mips(errno);

        FREE_TARGET_STRING(p, gpr[4]);

        break;

    case UHI_fstat:

        {

            struct stat sbuf;

            memset(&sbuf, 0, sizeof(sbuf));

            gpr[2] = fstat(gpr[4], &sbuf);

            gpr[3] = errno_mips(errno);

            if (gpr[2]) {

                goto uhi_done;

            }

            gpr[2] = copy_stat_to_target(env, &sbuf, gpr[5]);

            gpr[3] = errno_mips(errno);

        }

        break;

    case UHI_argc:

        gpr[2] = semihosting_get_argc();

        break;

    case UHI_argnlen:

        if (gpr[4] >= semihosting_get_argc()) {

            gpr[2] = -1;

            goto uhi_done;

        }

        gpr[2] = strlen(semihosting_get_arg(gpr[4]));

        break;

    case UHI_argn:

        if (gpr[4] >= semihosting_get_argc()) {

            gpr[2] = -1;

            goto uhi_done;

        }

        gpr[2] = copy_argn_to_target(env, gpr[4], gpr[5]);

        break;

    case UHI_plog:

        GET_TARGET_STRING(p, gpr[4]);

        p2 = strstr(p, "%d");

        if (p2) {

            int char_num = p2 - p;

            char *buf = g_malloc(char_num + 1);

            strncpy(buf, p, char_num);

            buf[char_num] = '\0';

            gpr[2] = printf("%s%d%s", buf, (int)gpr[5], p2 + 2);

            g_free(buf);

        } else {

            gpr[2] = printf("%s", p);

        }

        FREE_TARGET_STRING(p, gpr[4]);

        break;

    case UHI_assert:

        GET_TARGET_STRING(p, gpr[4]);

        GET_TARGET_STRING(p2, gpr[5]);

        printf("assertion '");

        printf("\"%s\"", p);

        printf("': file \"%s\", line %d\n", p2, (int)gpr[6]);

        FREE_TARGET_STRING(p2, gpr[5]);

        FREE_TARGET_STRING(p, gpr[4]);

        abort();

        break;

    case UHI_pread:

        gpr[2] = read_from_file(env, gpr[4], gpr[5], gpr[6], gpr[7]);

        gpr[3] = errno_mips(errno);

        break;

    case UHI_pwrite:

        gpr[2] = write_to_file(env, gpr[4], gpr[5], gpr[6], gpr[7]);

        gpr[3] = errno_mips(errno);

        break;

#ifndef _WIN32

    case UHI_link:

        GET_TARGET_STRING(p, gpr[4]);

        GET_TARGET_STRING(p2, gpr[5]);

        gpr[2] = link(p, p2);

        gpr[3] = errno_mips(errno);

        FREE_TARGET_STRING(p2, gpr[5]);

        FREE_TARGET_STRING(p, gpr[4]);

        break;

#endif

    default:

        fprintf(stderr, "Unknown UHI operation %d\n", op);

        abort();

    }

uhi_done:

    return;

}
