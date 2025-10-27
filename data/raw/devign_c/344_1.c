static void memory_dump(Monitor *mon, int count, int format, int wsize,

                        target_phys_addr_t addr, int is_physical)

{

    CPUState *env;

    int l, line_size, i, max_digits, len;

    uint8_t buf[16];

    uint64_t v;



    if (format == 'i') {

        int flags;

        flags = 0;

        env = mon_get_cpu();

        if (!env && !is_physical)

            return;

#ifdef TARGET_I386

        if (wsize == 2) {

            flags = 1;

        } else if (wsize == 4) {

            flags = 0;

        } else {

            /* as default we use the current CS size */

            flags = 0;

            if (env) {

#ifdef TARGET_X86_64

                if ((env->efer & MSR_EFER_LMA) &&

                    (env->segs[R_CS].flags & DESC_L_MASK))

                    flags = 2;

                else

#endif

                if (!(env->segs[R_CS].flags & DESC_B_MASK))

                    flags = 1;

            }

        }

#endif

        monitor_disas(mon, env, addr, count, is_physical, flags);

        return;

    }



    len = wsize * count;

    if (wsize == 1)

        line_size = 8;

    else

        line_size = 16;

    max_digits = 0;



    switch(format) {

    case 'o':

        max_digits = (wsize * 8 + 2) / 3;

        break;

    default:

    case 'x':

        max_digits = (wsize * 8) / 4;

        break;

    case 'u':

    case 'd':

        max_digits = (wsize * 8 * 10 + 32) / 33;

        break;

    case 'c':

        wsize = 1;

        break;

    }



    while (len > 0) {

        if (is_physical)

            monitor_printf(mon, TARGET_FMT_plx ":", addr);

        else

            monitor_printf(mon, TARGET_FMT_lx ":", (target_ulong)addr);

        l = len;

        if (l > line_size)

            l = line_size;

        if (is_physical) {

            cpu_physical_memory_rw(addr, buf, l, 0);

        } else {

            env = mon_get_cpu();

            if (!env)

                break;

            if (cpu_memory_rw_debug(env, addr, buf, l, 0) < 0) {

                monitor_printf(mon, " Cannot access memory\n");

                break;

            }

        }

        i = 0;

        while (i < l) {

            switch(wsize) {

            default:

            case 1:

                v = ldub_raw(buf + i);

                break;

            case 2:

                v = lduw_raw(buf + i);

                break;

            case 4:

                v = (uint32_t)ldl_raw(buf + i);

                break;

            case 8:

                v = ldq_raw(buf + i);

                break;

            }

            monitor_printf(mon, " ");

            switch(format) {

            case 'o':

                monitor_printf(mon, "%#*" PRIo64, max_digits, v);

                break;

            case 'x':

                monitor_printf(mon, "0x%0*" PRIx64, max_digits, v);

                break;

            case 'u':

                monitor_printf(mon, "%*" PRIu64, max_digits, v);

                break;

            case 'd':

                monitor_printf(mon, "%*" PRId64, max_digits, v);

                break;

            case 'c':

                monitor_printc(mon, v);

                break;

            }

            i += wsize;

        }

        monitor_printf(mon, "\n");

        addr += l;

        len -= l;

    }

}
