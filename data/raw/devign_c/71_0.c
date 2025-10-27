void gdb_do_syscall(gdb_syscall_complete_cb cb, const char *fmt, ...)

{

    va_list va;

    char buf[256];

    char *p;

    target_ulong addr;

    uint64_t i64;

    GDBState *s;



    s = gdbserver_state;

    if (!s)

        return;

    gdb_current_syscall_cb = cb;

    s->state = RS_SYSCALL;

#ifndef CONFIG_USER_ONLY

    vm_stop(EXCP_DEBUG);

#endif

    s->state = RS_IDLE;

    va_start(va, fmt);

    p = buf;

    *(p++) = 'F';

    while (*fmt) {

        if (*fmt == '%') {

            fmt++;

            switch (*fmt++) {

            case 'x':

                addr = va_arg(va, target_ulong);

                p += snprintf(p, &buf[sizeof(buf)] - p, TARGET_FMT_lx, addr);

                break;

            case 'l':

                if (*(fmt++) != 'x')

                    goto bad_format;

                i64 = va_arg(va, uint64_t);

                p += snprintf(p, &buf[sizeof(buf)] - p, "%" PRIx64, i64);

                break;

            case 's':

                addr = va_arg(va, target_ulong);

                p += snprintf(p, &buf[sizeof(buf)] - p, TARGET_FMT_lx "/%x",

                              addr, va_arg(va, int));

                break;

            default:

            bad_format:

                fprintf(stderr, "gdbstub: Bad syscall format string '%s'\n",

                        fmt - 1);

                break;

            }

        } else {

            *(p++) = *(fmt++);

        }

    }

    *p = 0;

    va_end(va);

    put_packet(s, buf);

#ifdef CONFIG_USER_ONLY

    gdb_handlesig(s->c_cpu, 0);

#else

    cpu_interrupt(s->c_cpu, CPU_INTERRUPT_EXIT);

#endif

}
