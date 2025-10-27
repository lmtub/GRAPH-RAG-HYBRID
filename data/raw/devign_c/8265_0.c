static const mon_cmd_t *monitor_parse_command(Monitor *mon,

                                              const char *cmdline,

                                              int start,

                                              mon_cmd_t *table,

                                              QDict *qdict)

{

    const char *p, *typestr;

    int c;

    const mon_cmd_t *cmd;

    char cmdname[256];

    char buf[1024];

    char *key;



    /* extract the command name */

    p = get_command_name(cmdline + start, cmdname, sizeof(cmdname));

    if (!p)

        return NULL;



    cmd = search_dispatch_table(table, cmdname);

    if (!cmd) {

        monitor_printf(mon, "unknown command: '%.*s'\n",

                       (int)(p - cmdline), cmdline);

        return NULL;

    }



    /* filter out following useless space */

    while (qemu_isspace(*p)) {

        p++;

    }

    /* search sub command */

    if (cmd->sub_table != NULL) {

        /* check if user set additional command */

        if (*p == '\0') {

            return cmd;

        }

        return monitor_parse_command(mon, cmdline, p - cmdline,

                                     cmd->sub_table, qdict);

    }



    /* parse the parameters */

    typestr = cmd->args_type;

    for(;;) {

        typestr = key_get_info(typestr, &key);

        if (!typestr)

            break;

        c = *typestr;

        typestr++;

        switch(c) {

        case 'F':

        case 'B':

        case 's':

            {

                int ret;



                while (qemu_isspace(*p))

                    p++;

                if (*typestr == '?') {

                    typestr++;

                    if (*p == '\0') {

                        /* no optional string: NULL argument */

                        break;

                    }

                }

                ret = get_str(buf, sizeof(buf), &p);

                if (ret < 0) {

                    switch(c) {

                    case 'F':

                        monitor_printf(mon, "%s: filename expected\n",

                                       cmdname);

                        break;

                    case 'B':

                        monitor_printf(mon, "%s: block device name expected\n",

                                       cmdname);

                        break;

                    default:

                        monitor_printf(mon, "%s: string expected\n", cmdname);

                        break;

                    }

                    goto fail;

                }

                qdict_put(qdict, key, qstring_from_str(buf));

            }

            break;

        case 'O':

            {

                QemuOptsList *opts_list;

                QemuOpts *opts;



                opts_list = qemu_find_opts(key);

                if (!opts_list || opts_list->desc->name) {

                    goto bad_type;

                }

                while (qemu_isspace(*p)) {

                    p++;

                }

                if (!*p)

                    break;

                if (get_str(buf, sizeof(buf), &p) < 0) {

                    goto fail;

                }

                opts = qemu_opts_parse(opts_list, buf, 1);

                if (!opts) {

                    goto fail;

                }

                qemu_opts_to_qdict(opts, qdict);

                qemu_opts_del(opts);

            }

            break;

        case '/':

            {

                int count, format, size;



                while (qemu_isspace(*p))

                    p++;

                if (*p == '/') {

                    /* format found */

                    p++;

                    count = 1;

                    if (qemu_isdigit(*p)) {

                        count = 0;

                        while (qemu_isdigit(*p)) {

                            count = count * 10 + (*p - '0');

                            p++;

                        }

                    }

                    size = -1;

                    format = -1;

                    for(;;) {

                        switch(*p) {

                        case 'o':

                        case 'd':

                        case 'u':

                        case 'x':

                        case 'i':

                        case 'c':

                            format = *p++;

                            break;

                        case 'b':

                            size = 1;

                            p++;

                            break;

                        case 'h':

                            size = 2;

                            p++;

                            break;

                        case 'w':

                            size = 4;

                            p++;

                            break;

                        case 'g':

                        case 'L':

                            size = 8;

                            p++;

                            break;

                        default:

                            goto next;

                        }

                    }

                next:

                    if (*p != '\0' && !qemu_isspace(*p)) {

                        monitor_printf(mon, "invalid char in format: '%c'\n",

                                       *p);

                        goto fail;

                    }

                    if (format < 0)

                        format = default_fmt_format;

                    if (format != 'i') {

                        /* for 'i', not specifying a size gives -1 as size */

                        if (size < 0)

                            size = default_fmt_size;

                        default_fmt_size = size;

                    }

                    default_fmt_format = format;

                } else {

                    count = 1;

                    format = default_fmt_format;

                    if (format != 'i') {

                        size = default_fmt_size;

                    } else {

                        size = -1;

                    }

                }

                qdict_put(qdict, "count", qint_from_int(count));

                qdict_put(qdict, "format", qint_from_int(format));

                qdict_put(qdict, "size", qint_from_int(size));

            }

            break;

        case 'i':

        case 'l':

        case 'M':

            {

                int64_t val;



                while (qemu_isspace(*p))

                    p++;

                if (*typestr == '?' || *typestr == '.') {

                    if (*typestr == '?') {

                        if (*p == '\0') {

                            typestr++;

                            break;

                        }

                    } else {

                        if (*p == '.') {

                            p++;

                            while (qemu_isspace(*p))

                                p++;

                        } else {

                            typestr++;

                            break;

                        }

                    }

                    typestr++;

                }

                if (get_expr(mon, &val, &p))

                    goto fail;

                /* Check if 'i' is greater than 32-bit */

                if ((c == 'i') && ((val >> 32) & 0xffffffff)) {

                    monitor_printf(mon, "\'%s\' has failed: ", cmdname);

                    monitor_printf(mon, "integer is for 32-bit values\n");

                    goto fail;

                } else if (c == 'M') {

                    if (val < 0) {

                        monitor_printf(mon, "enter a positive value\n");

                        goto fail;

                    }

                    val <<= 20;

                }

                qdict_put(qdict, key, qint_from_int(val));

            }

            break;

        case 'o':

            {

                int64_t val;

                char *end;



                while (qemu_isspace(*p)) {

                    p++;

                }

                if (*typestr == '?') {

                    typestr++;

                    if (*p == '\0') {

                        break;

                    }

                }

                val = strtosz(p, &end);

                if (val < 0) {

                    monitor_printf(mon, "invalid size\n");

                    goto fail;

                }

                qdict_put(qdict, key, qint_from_int(val));

                p = end;

            }

            break;

        case 'T':

            {

                double val;



                while (qemu_isspace(*p))

                    p++;

                if (*typestr == '?') {

                    typestr++;

                    if (*p == '\0') {

                        break;

                    }

                }

                if (get_double(mon, &val, &p) < 0) {

                    goto fail;

                }

                if (p[0] && p[1] == 's') {

                    switch (*p) {

                    case 'm':

                        val /= 1e3; p += 2; break;

                    case 'u':

                        val /= 1e6; p += 2; break;

                    case 'n':

                        val /= 1e9; p += 2; break;

                    }

                }

                if (*p && !qemu_isspace(*p)) {

                    monitor_printf(mon, "Unknown unit suffix\n");

                    goto fail;

                }

                qdict_put(qdict, key, qfloat_from_double(val));

            }

            break;

        case 'b':

            {

                const char *beg;

                bool val;



                while (qemu_isspace(*p)) {

                    p++;

                }

                beg = p;

                while (qemu_isgraph(*p)) {

                    p++;

                }

                if (p - beg == 2 && !memcmp(beg, "on", p - beg)) {

                    val = true;

                } else if (p - beg == 3 && !memcmp(beg, "off", p - beg)) {

                    val = false;

                } else {

                    monitor_printf(mon, "Expected 'on' or 'off'\n");

                    goto fail;

                }

                qdict_put(qdict, key, qbool_from_bool(val));

            }

            break;

        case '-':

            {

                const char *tmp = p;

                int skip_key = 0;

                /* option */



                c = *typestr++;

                if (c == '\0')

                    goto bad_type;

                while (qemu_isspace(*p))

                    p++;

                if (*p == '-') {

                    p++;

                    if(c != *p) {

                        if(!is_valid_option(p, typestr)) {

                  

                            monitor_printf(mon, "%s: unsupported option -%c\n",

                                           cmdname, *p);

                            goto fail;

                        } else {

                            skip_key = 1;

                        }

                    }

                    if(skip_key) {

                        p = tmp;

                    } else {

                        /* has option */

                        p++;

                        qdict_put(qdict, key, qbool_from_bool(true));

                    }

                }

            }

            break;

        case 'S':

            {

                /* package all remaining string */

                int len;



                while (qemu_isspace(*p)) {

                    p++;

                }

                if (*typestr == '?') {

                    typestr++;

                    if (*p == '\0') {

                        /* no remaining string: NULL argument */

                        break;

                    }

                }

                len = strlen(p);

                if (len <= 0) {

                    monitor_printf(mon, "%s: string expected\n",

                                   cmdname);

                    break;

                }

                qdict_put(qdict, key, qstring_from_str(p));

                p += len;

            }

            break;

        default:

        bad_type:

            monitor_printf(mon, "%s: unknown type '%c'\n", cmdname, c);

            goto fail;

        }

        g_free(key);

        key = NULL;

    }

    /* check that all arguments were parsed */

    while (qemu_isspace(*p))

        p++;

    if (*p != '\0') {

        monitor_printf(mon, "%s: extraneous characters at the end of line\n",

                       cmdname);

        goto fail;

    }



    return cmd;



fail:

    g_free(key);

    return NULL;

}
