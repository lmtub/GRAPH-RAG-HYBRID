static int gdb_handle_packet(GDBState *s, CPUState *env, const char *line_buf)

{

    const char *p;

    int ch, reg_size, type;

    char buf[4096];

    uint8_t mem_buf[4096];

    uint32_t *registers;

    target_ulong addr, len;



#ifdef DEBUG_GDB

    printf("command='%s'\n", line_buf);

#endif

    p = line_buf;

    ch = *p++;

    switch(ch) {

    case '?':

        /* TODO: Make this return the correct value for user-mode.  */

        snprintf(buf, sizeof(buf), "S%02x", SIGTRAP);

        put_packet(s, buf);

        /* Remove all the breakpoints when this query is issued,

         * because gdb is doing and initial connect and the state

         * should be cleaned up.

         */




    case 'c':

        if (*p != '\0') {

            addr = strtoull(p, (char **)&p, 16);

#if defined(TARGET_I386)

            env->eip = addr;

#elif defined (TARGET_PPC)

            env->nip = addr;

#elif defined (TARGET_SPARC)

            env->pc = addr;

            env->npc = addr + 4;

#elif defined (TARGET_ARM)

            env->regs[15] = addr;

#elif defined (TARGET_SH4)

            env->pc = addr;

#elif defined (TARGET_MIPS)

            env->PC[env->current_tc] = addr;

#elif defined (TARGET_CRIS)

            env->pc = addr;

#endif

        }


	return RS_IDLE;












    case 's':

        if (*p != '\0') {

            addr = strtoull(p, (char **)&p, 16);

#if defined(TARGET_I386)

            env->eip = addr;

#elif defined (TARGET_PPC)

            env->nip = addr;

#elif defined (TARGET_SPARC)

            env->pc = addr;

            env->npc = addr + 4;

#elif defined (TARGET_ARM)

            env->regs[15] = addr;

#elif defined (TARGET_SH4)

            env->pc = addr;

#elif defined (TARGET_MIPS)

            env->PC[env->current_tc] = addr;

#elif defined (TARGET_CRIS)

            env->pc = addr;

#endif

        }

        cpu_single_step(env, sstep_flags);


	return RS_IDLE;

    case 'F':

        {

            target_ulong ret;

            target_ulong err;



            ret = strtoull(p, (char **)&p, 16);

            if (*p == ',') {

                p++;

                err = strtoull(p, (char **)&p, 16);

            } else {

                err = 0;

            }

            if (*p == ',')

                p++;

            type = *p;

            if (gdb_current_syscall_cb)

                gdb_current_syscall_cb(s->env, ret, err);

            if (type == 'C') {

                put_packet(s, "T02");

            } else {


            }

        }


    case 'g':

        reg_size = cpu_gdb_read_registers(env, mem_buf);

        memtohex(buf, mem_buf, reg_size);

        put_packet(s, buf);


    case 'G':

        registers = (void *)mem_buf;

        len = strlen(p) / 2;

        hextomem((uint8_t *)registers, p, len);

        cpu_gdb_write_registers(env, mem_buf, len);



    case 'm':

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, NULL, 16);

        if (cpu_memory_rw_debug(env, addr, mem_buf, len, 0) != 0) {

            put_packet (s, "E14");

        } else {

            memtohex(buf, mem_buf, len);

            put_packet(s, buf);

        }


    case 'M':

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, (char **)&p, 16);

        if (*p == ':')

            p++;

        hextomem(mem_buf, p, len);

        if (cpu_memory_rw_debug(env, addr, mem_buf, len, 1) != 0)

            put_packet(s, "E14");

        else



    case 'Z':

        type = strtoul(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, (char **)&p, 16);

        if (type == 0 || type == 1) {

            if (cpu_breakpoint_insert(env, addr) < 0)

                goto breakpoint_error;


#ifndef CONFIG_USER_ONLY

        } else if (type == 2) {

            if (cpu_watchpoint_insert(env, addr) < 0)

                goto breakpoint_error;


#endif

        } else {

        breakpoint_error:

            put_packet(s, "E22");

        }


    case 'z':

        type = strtoul(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        addr = strtoull(p, (char **)&p, 16);

        if (*p == ',')

            p++;

        len = strtoull(p, (char **)&p, 16);

        if (type == 0 || type == 1) {

            cpu_breakpoint_remove(env, addr);


#ifndef CONFIG_USER_ONLY

        } else if (type == 2) {

            cpu_watchpoint_remove(env, addr);


#endif

        } else {

            goto breakpoint_error;

        }


    case 'q':

    case 'Q':

        /* parse any 'q' packets here */

        if (!strcmp(p,"qemu.sstepbits")) {

            /* Query Breakpoint bit definitions */

            sprintf(buf,"ENABLE=%x,NOIRQ=%x,NOTIMER=%x",

                    SSTEP_ENABLE,

                    SSTEP_NOIRQ,

                    SSTEP_NOTIMER);

            put_packet(s, buf);


        } else if (strncmp(p,"qemu.sstep",10) == 0) {

            /* Display or change the sstep_flags */

            p += 10;

            if (*p != '=') {

                /* Display current setting */

                sprintf(buf,"0x%x", sstep_flags);

                put_packet(s, buf);


            }

            p++;

            type = strtoul(p, (char **)&p, 16);

            sstep_flags = type;



        }

#ifdef CONFIG_LINUX_USER

        else if (strncmp(p, "Offsets", 7) == 0) {

            TaskState *ts = env->opaque;



            sprintf(buf,

                    "Text=" TARGET_ABI_FMT_lx ";Data=" TARGET_ABI_FMT_lx

                    ";Bss=" TARGET_ABI_FMT_lx,

                    ts->info->code_offset,

                    ts->info->data_offset,

                    ts->info->data_offset);

            put_packet(s, buf);


        }

#endif

        /* Fall through.  */

    default:

        /* put empty packet */

        buf[0] = '\0';

        put_packet(s, buf);


    }

    return RS_IDLE;

}