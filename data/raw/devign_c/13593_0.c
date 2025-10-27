int main(int argc, char **argv, char **envp)

{

    const char *gdbstub_dev = NULL;

    int i;

    int snapshot, linux_boot;

    const char *icount_option = NULL;

    const char *initrd_filename;

    const char *kernel_filename, *kernel_cmdline;

    char boot_devices[33] = "cad"; /* default to HD->floppy->CD-ROM */

    DisplayState *ds;

    DisplayChangeListener *dcl;

    int cyls, heads, secs, translation;

    QemuOpts *hda_opts = NULL, *opts;

    QemuOptsList *olist;

    int optind;

    const char *optarg;

    const char *loadvm = NULL;

    QEMUMachine *machine;

    const char *cpu_model;

    const char *pid_file = NULL;

    const char *incoming = NULL;

#ifdef CONFIG_VNC

    int show_vnc_port = 0;

#endif

    int defconfig = 1;

    const char *log_mask = NULL;

    const char *log_file = NULL;

    GMemVTable mem_trace = {

        .malloc = malloc_and_trace,

        .realloc = realloc_and_trace,

        .free = free_and_trace,

    };

    const char *trace_events = NULL;

    const char *trace_file = NULL;



    atexit(qemu_run_exit_notifiers);

    error_set_progname(argv[0]);



    g_mem_set_vtable(&mem_trace);

    if (!g_thread_supported()) {

#if !GLIB_CHECK_VERSION(2, 31, 0)

        g_thread_init(NULL);

#else

        fprintf(stderr, "glib threading failed to initialize.\n");

        exit(1);

#endif

    }



    runstate_init();



    init_clocks();

    rtc_clock = host_clock;



    qemu_cache_utils_init(envp);



    QLIST_INIT (&vm_change_state_head);

    os_setup_early_signal_handling();



    module_call_init(MODULE_INIT_MACHINE);

    machine = find_default_machine();

    cpu_model = NULL;

    initrd_filename = NULL;

    ram_size = 0;

    snapshot = 0;

    kernel_filename = NULL;

    kernel_cmdline = "";

    cyls = heads = secs = 0;

    translation = BIOS_ATA_TRANSLATION_AUTO;



    for (i = 0; i < MAX_NODES; i++) {

        node_mem[i] = 0;

        node_cpumask[i] = 0;

    }



    nb_numa_nodes = 0;

    nb_nics = 0;



    autostart= 1;



    /* first pass of option parsing */

    optind = 1;

    while (optind < argc) {

        if (argv[optind][0] != '-') {

            /* disk image */

            optind++;

            continue;

        } else {

            const QEMUOption *popt;



            popt = lookup_opt(argc, argv, &optarg, &optind);

            switch (popt->index) {

            case QEMU_OPTION_nodefconfig:

                defconfig=0;

                break;

            }

        }

    }



    if (defconfig) {

        int ret;



        ret = qemu_read_config_file(CONFIG_QEMU_CONFDIR "/qemu.conf");

        if (ret < 0 && ret != -ENOENT) {

            exit(1);

        }



        ret = qemu_read_config_file(arch_config_name);

        if (ret < 0 && ret != -ENOENT) {

            exit(1);

        }

    }

    cpudef_init();



    /* second pass of option parsing */

    optind = 1;

    for(;;) {

        if (optind >= argc)

            break;

        if (argv[optind][0] != '-') {

	    hda_opts = drive_add(IF_DEFAULT, 0, argv[optind++], HD_OPTS);

        } else {

            const QEMUOption *popt;



            popt = lookup_opt(argc, argv, &optarg, &optind);

            if (!(popt->arch_mask & arch_type)) {

                printf("Option %s not supported for this target\n", popt->name);

                exit(1);

            }

            switch(popt->index) {

            case QEMU_OPTION_M:

                machine = machine_parse(optarg);

                break;

            case QEMU_OPTION_cpu:

                /* hw initialization will check this */

                if (*optarg == '?') {

                    list_cpus(stdout, &fprintf, optarg);

                    exit(0);

                } else {

                    cpu_model = optarg;

                }

                break;

            case QEMU_OPTION_initrd:

                initrd_filename = optarg;

                break;

            case QEMU_OPTION_hda:

                {

                    char buf[256];

                    if (cyls == 0)

                        snprintf(buf, sizeof(buf), "%s", HD_OPTS);

                    else

                        snprintf(buf, sizeof(buf),

                                 "%s,cyls=%d,heads=%d,secs=%d%s",

                                 HD_OPTS , cyls, heads, secs,

                                 translation == BIOS_ATA_TRANSLATION_LBA ?

                                 ",trans=lba" :

                                 translation == BIOS_ATA_TRANSLATION_NONE ?

                                 ",trans=none" : "");

                    drive_add(IF_DEFAULT, 0, optarg, buf);

                    break;

                }

            case QEMU_OPTION_hdb:

            case QEMU_OPTION_hdc:

            case QEMU_OPTION_hdd:

                drive_add(IF_DEFAULT, popt->index - QEMU_OPTION_hda, optarg,

                          HD_OPTS);

                break;

            case QEMU_OPTION_drive:

                if (drive_def(optarg) == NULL) {

                    exit(1);

                }

	        break;

            case QEMU_OPTION_set:

                if (qemu_set_option(optarg) != 0)

                    exit(1);

	        break;

            case QEMU_OPTION_global:

                if (qemu_global_option(optarg) != 0)

                    exit(1);

	        break;

            case QEMU_OPTION_mtdblock:

                drive_add(IF_MTD, -1, optarg, MTD_OPTS);

                break;

            case QEMU_OPTION_sd:

                drive_add(IF_SD, 0, optarg, SD_OPTS);

                break;

            case QEMU_OPTION_pflash:

                drive_add(IF_PFLASH, -1, optarg, PFLASH_OPTS);

                break;

            case QEMU_OPTION_snapshot:

                snapshot = 1;

                break;

            case QEMU_OPTION_hdachs:

                {

                    const char *p;

                    p = optarg;

                    cyls = strtol(p, (char **)&p, 0);

                    if (cyls < 1 || cyls > 16383)

                        goto chs_fail;

                    if (*p != ',')

                        goto chs_fail;

                    p++;

                    heads = strtol(p, (char **)&p, 0);

                    if (heads < 1 || heads > 16)

                        goto chs_fail;

                    if (*p != ',')

                        goto chs_fail;

                    p++;

                    secs = strtol(p, (char **)&p, 0);

                    if (secs < 1 || secs > 63)

                        goto chs_fail;

                    if (*p == ',') {

                        p++;

                        if (!strcmp(p, "none"))

                            translation = BIOS_ATA_TRANSLATION_NONE;

                        else if (!strcmp(p, "lba"))

                            translation = BIOS_ATA_TRANSLATION_LBA;

                        else if (!strcmp(p, "auto"))

                            translation = BIOS_ATA_TRANSLATION_AUTO;

                        else

                            goto chs_fail;

                    } else if (*p != '\0') {

                    chs_fail:

                        fprintf(stderr, "qemu: invalid physical CHS format\n");

                        exit(1);

                    }

		    if (hda_opts != NULL) {

                        char num[16];

                        snprintf(num, sizeof(num), "%d", cyls);

                        qemu_opt_set(hda_opts, "cyls", num);

                        snprintf(num, sizeof(num), "%d", heads);

                        qemu_opt_set(hda_opts, "heads", num);

                        snprintf(num, sizeof(num), "%d", secs);

                        qemu_opt_set(hda_opts, "secs", num);

                        if (translation == BIOS_ATA_TRANSLATION_LBA)

                            qemu_opt_set(hda_opts, "trans", "lba");

                        if (translation == BIOS_ATA_TRANSLATION_NONE)

                            qemu_opt_set(hda_opts, "trans", "none");

                    }

                }

                break;

            case QEMU_OPTION_numa:

                if (nb_numa_nodes >= MAX_NODES) {

                    fprintf(stderr, "qemu: too many NUMA nodes\n");

                    exit(1);

                }

                numa_add(optarg);

                break;

            case QEMU_OPTION_display:

                display_type = select_display(optarg);

                break;

            case QEMU_OPTION_nographic:

                display_type = DT_NOGRAPHIC;

                break;

            case QEMU_OPTION_curses:

#ifdef CONFIG_CURSES

                display_type = DT_CURSES;

#else

                fprintf(stderr, "Curses support is disabled\n");

                exit(1);

#endif

                break;

            case QEMU_OPTION_portrait:

                graphic_rotate = 90;

                break;

            case QEMU_OPTION_rotate:

                graphic_rotate = strtol(optarg, (char **) &optarg, 10);

                if (graphic_rotate != 0 && graphic_rotate != 90 &&

                    graphic_rotate != 180 && graphic_rotate != 270) {

                    fprintf(stderr,

                        "qemu: only 90, 180, 270 deg rotation is available\n");

                    exit(1);

                }

                break;

            case QEMU_OPTION_kernel:

                kernel_filename = optarg;

                break;

            case QEMU_OPTION_append:

                kernel_cmdline = optarg;

                break;

            case QEMU_OPTION_cdrom:

                drive_add(IF_DEFAULT, 2, optarg, CDROM_OPTS);

                break;

            case QEMU_OPTION_boot:

                {

                    static const char * const params[] = {

                        "order", "once", "menu",

                        "splash", "splash-time", NULL

                    };

                    char buf[sizeof(boot_devices)];

                    char *standard_boot_devices;

                    int legacy = 0;



                    if (!strchr(optarg, '=')) {

                        legacy = 1;

                        pstrcpy(buf, sizeof(buf), optarg);

                    } else if (check_params(buf, sizeof(buf), params, optarg) < 0) {

                        fprintf(stderr,

                                "qemu: unknown boot parameter '%s' in '%s'\n",

                                buf, optarg);

                        exit(1);

                    }



                    if (legacy ||

                        get_param_value(buf, sizeof(buf), "order", optarg)) {

                        validate_bootdevices(buf);

                        pstrcpy(boot_devices, sizeof(boot_devices), buf);

                    }

                    if (!legacy) {

                        if (get_param_value(buf, sizeof(buf),

                                            "once", optarg)) {

                            validate_bootdevices(buf);

                            standard_boot_devices = g_strdup(boot_devices);

                            pstrcpy(boot_devices, sizeof(boot_devices), buf);

                            qemu_register_reset(restore_boot_devices,

                                                standard_boot_devices);

                        }

                        if (get_param_value(buf, sizeof(buf),

                                            "menu", optarg)) {

                            if (!strcmp(buf, "on")) {

                                boot_menu = 1;

                            } else if (!strcmp(buf, "off")) {

                                boot_menu = 0;

                            } else {

                                fprintf(stderr,

                                        "qemu: invalid option value '%s'\n",

                                        buf);

                                exit(1);

                            }

                        }

                        qemu_opts_parse(qemu_find_opts("boot-opts"),

                                        optarg, 0);

                    }

                }

                break;

            case QEMU_OPTION_fda:

            case QEMU_OPTION_fdb:

                drive_add(IF_FLOPPY, popt->index - QEMU_OPTION_fda,

                          optarg, FD_OPTS);

                break;

            case QEMU_OPTION_no_fd_bootchk:

                fd_bootchk = 0;

                break;

            case QEMU_OPTION_netdev:

                if (net_client_parse(qemu_find_opts("netdev"), optarg) == -1) {

                    exit(1);

                }

                break;

            case QEMU_OPTION_net:

                if (net_client_parse(qemu_find_opts("net"), optarg) == -1) {

                    exit(1);

                }

                break;

#ifdef CONFIG_SLIRP

            case QEMU_OPTION_tftp:

                legacy_tftp_prefix = optarg;

                break;

            case QEMU_OPTION_bootp:

                legacy_bootp_filename = optarg;

                break;

            case QEMU_OPTION_redir:

                if (net_slirp_redir(optarg) < 0)

                    exit(1);

                break;

#endif

            case QEMU_OPTION_bt:

                add_device_config(DEV_BT, optarg);

                break;

            case QEMU_OPTION_audio_help:

                if (!(audio_available())) {

                    printf("Option %s not supported for this target\n", popt->name);

                    exit(1);

                }

                AUD_help ();

                exit (0);

                break;

            case QEMU_OPTION_soundhw:

                if (!(audio_available())) {

                    printf("Option %s not supported for this target\n", popt->name);

                    exit(1);

                }

                select_soundhw (optarg);

                break;

            case QEMU_OPTION_h:

                help(0);

                break;

            case QEMU_OPTION_version:

                version();

                exit(0);

                break;

            case QEMU_OPTION_m: {

                int64_t value;

                char *end;



                value = strtosz(optarg, &end);

                if (value < 0 || *end) {

                    fprintf(stderr, "qemu: invalid ram size: %s\n", optarg);

                    exit(1);

                }



                if (value != (uint64_t)(ram_addr_t)value) {

                    fprintf(stderr, "qemu: ram size too large\n");

                    exit(1);

                }

                ram_size = value;

                break;

            }

            case QEMU_OPTION_mempath:

                mem_path = optarg;

                break;

#ifdef MAP_POPULATE

            case QEMU_OPTION_mem_prealloc:

                mem_prealloc = 1;

                break;

#endif

            case QEMU_OPTION_d:

                log_mask = optarg;

                break;

            case QEMU_OPTION_D:

                log_file = optarg;

                break;

            case QEMU_OPTION_s:

                gdbstub_dev = "tcp::" DEFAULT_GDBSTUB_PORT;

                break;

            case QEMU_OPTION_gdb:

                gdbstub_dev = optarg;

                break;

            case QEMU_OPTION_L:

                data_dir = optarg;

                break;

            case QEMU_OPTION_bios:

                bios_name = optarg;

                break;

            case QEMU_OPTION_singlestep:

                singlestep = 1;

                break;

            case QEMU_OPTION_S:

                autostart = 0;

                break;

	    case QEMU_OPTION_k:

		keyboard_layout = optarg;

		break;

            case QEMU_OPTION_localtime:

                rtc_utc = 0;

                break;

            case QEMU_OPTION_vga:

                select_vgahw (optarg);

                break;

            case QEMU_OPTION_g:

                {

                    const char *p;

                    int w, h, depth;

                    p = optarg;

                    w = strtol(p, (char **)&p, 10);

                    if (w <= 0) {

                    graphic_error:

                        fprintf(stderr, "qemu: invalid resolution or depth\n");

                        exit(1);

                    }

                    if (*p != 'x')

                        goto graphic_error;

                    p++;

                    h = strtol(p, (char **)&p, 10);

                    if (h <= 0)

                        goto graphic_error;

                    if (*p == 'x') {

                        p++;

                        depth = strtol(p, (char **)&p, 10);

                        if (depth != 8 && depth != 15 && depth != 16 &&

                            depth != 24 && depth != 32)

                            goto graphic_error;

                    } else if (*p == '\0') {

                        depth = graphic_depth;

                    } else {

                        goto graphic_error;

                    }



                    graphic_width = w;

                    graphic_height = h;

                    graphic_depth = depth;

                }

                break;

            case QEMU_OPTION_echr:

                {

                    char *r;

                    term_escape_char = strtol(optarg, &r, 0);

                    if (r == optarg)

                        printf("Bad argument to echr\n");

                    break;

                }

            case QEMU_OPTION_monitor:

                monitor_parse(optarg, "readline");

                default_monitor = 0;

                break;

            case QEMU_OPTION_qmp:

                monitor_parse(optarg, "control");

                default_monitor = 0;

                break;

            case QEMU_OPTION_mon:

                opts = qemu_opts_parse(qemu_find_opts("mon"), optarg, 1);

                if (!opts) {

                    exit(1);

                }

                default_monitor = 0;

                break;

            case QEMU_OPTION_chardev:

                opts = qemu_opts_parse(qemu_find_opts("chardev"), optarg, 1);

                if (!opts) {

                    exit(1);

                }

                break;

            case QEMU_OPTION_fsdev:

                olist = qemu_find_opts("fsdev");

                if (!olist) {

                    fprintf(stderr, "fsdev is not supported by this qemu build.\n");

                    exit(1);

                }

                opts = qemu_opts_parse(olist, optarg, 1);

                if (!opts) {

                    fprintf(stderr, "parse error: %s\n", optarg);

                    exit(1);

                }

                break;

            case QEMU_OPTION_virtfs: {

                QemuOpts *fsdev;

                QemuOpts *device;

                const char *writeout;



                olist = qemu_find_opts("virtfs");

                if (!olist) {

                    fprintf(stderr, "virtfs is not supported by this qemu build.\n");

                    exit(1);

                }

                opts = qemu_opts_parse(olist, optarg, 1);

                if (!opts) {

                    fprintf(stderr, "parse error: %s\n", optarg);

                    exit(1);

                }



                if (qemu_opt_get(opts, "fsdriver") == NULL ||

                        qemu_opt_get(opts, "mount_tag") == NULL ||

                        qemu_opt_get(opts, "path") == NULL) {

                    fprintf(stderr, "Usage: -virtfs fsdriver,path=/share_path/,"

                            "[security_model={mapped|passthrough|none}],"

                            "mount_tag=tag.\n");

                    exit(1);

                }

                fsdev = qemu_opts_create(qemu_find_opts("fsdev"),

                                         qemu_opt_get(opts, "mount_tag"), 1);

                if (!fsdev) {

                    fprintf(stderr, "duplicate fsdev id: %s\n",

                            qemu_opt_get(opts, "mount_tag"));

                    exit(1);

                }



                writeout = qemu_opt_get(opts, "writeout");

                if (writeout) {

#ifdef CONFIG_SYNC_FILE_RANGE

                    qemu_opt_set(fsdev, "writeout", writeout);

#else

                    fprintf(stderr, "writeout=immediate not supported on "

                            "this platform\n");

                    exit(1);

#endif

                }

                qemu_opt_set(fsdev, "fsdriver", qemu_opt_get(opts, "fsdriver"));

                qemu_opt_set(fsdev, "path", qemu_opt_get(opts, "path"));

                qemu_opt_set(fsdev, "security_model",

                             qemu_opt_get(opts, "security_model"));



                qemu_opt_set_bool(fsdev, "readonly",

                                qemu_opt_get_bool(opts, "readonly", 0));

                device = qemu_opts_create(qemu_find_opts("device"), NULL, 0);

                qemu_opt_set(device, "driver", "virtio-9p-pci");

                qemu_opt_set(device, "fsdev",

                             qemu_opt_get(opts, "mount_tag"));

                qemu_opt_set(device, "mount_tag",

                             qemu_opt_get(opts, "mount_tag"));

                break;

            }

            case QEMU_OPTION_virtfs_synth: {

                QemuOpts *fsdev;

                QemuOpts *device;



                fsdev = qemu_opts_create(qemu_find_opts("fsdev"), "v_synth", 1);

                if (!fsdev) {

                    fprintf(stderr, "duplicate option: %s\n", "virtfs_synth");

                    exit(1);

                }

                qemu_opt_set(fsdev, "fsdriver", "synth");

                qemu_opt_set(fsdev, "path", "/"); /* ignored */



                device = qemu_opts_create(qemu_find_opts("device"), NULL, 0);

                qemu_opt_set(device, "driver", "virtio-9p-pci");

                qemu_opt_set(device, "fsdev", "v_synth");

                qemu_opt_set(device, "mount_tag", "v_synth");

                break;

            }

            case QEMU_OPTION_serial:

                add_device_config(DEV_SERIAL, optarg);

                default_serial = 0;

                if (strncmp(optarg, "mon:", 4) == 0) {

                    default_monitor = 0;

                }

                break;

            case QEMU_OPTION_watchdog:

                if (watchdog) {

                    fprintf(stderr,

                            "qemu: only one watchdog option may be given\n");

                    return 1;

                }

                watchdog = optarg;

                break;

            case QEMU_OPTION_watchdog_action:

                if (select_watchdog_action(optarg) == -1) {

                    fprintf(stderr, "Unknown -watchdog-action parameter\n");

                    exit(1);

                }

                break;

            case QEMU_OPTION_virtiocon:

                add_device_config(DEV_VIRTCON, optarg);

                default_virtcon = 0;

                if (strncmp(optarg, "mon:", 4) == 0) {

                    default_monitor = 0;

                }

                break;

            case QEMU_OPTION_parallel:

                add_device_config(DEV_PARALLEL, optarg);

                default_parallel = 0;

                if (strncmp(optarg, "mon:", 4) == 0) {

                    default_monitor = 0;

                }

                break;

            case QEMU_OPTION_debugcon:

                add_device_config(DEV_DEBUGCON, optarg);

                break;

	    case QEMU_OPTION_loadvm:

		loadvm = optarg;

		break;

            case QEMU_OPTION_full_screen:

                full_screen = 1;

                break;

#ifdef CONFIG_SDL

            case QEMU_OPTION_no_frame:

                no_frame = 1;

                break;

            case QEMU_OPTION_alt_grab:

                alt_grab = 1;

                break;

            case QEMU_OPTION_ctrl_grab:

                ctrl_grab = 1;

                break;

            case QEMU_OPTION_no_quit:

                no_quit = 1;

                break;

            case QEMU_OPTION_sdl:

                display_type = DT_SDL;

                break;

#else

            case QEMU_OPTION_no_frame:

            case QEMU_OPTION_alt_grab:

            case QEMU_OPTION_ctrl_grab:

            case QEMU_OPTION_no_quit:

            case QEMU_OPTION_sdl:

                fprintf(stderr, "SDL support is disabled\n");

                exit(1);

#endif

            case QEMU_OPTION_pidfile:

                pid_file = optarg;

                break;

            case QEMU_OPTION_win2k_hack:

                win2k_install_hack = 1;

                break;

            case QEMU_OPTION_rtc_td_hack:

                rtc_td_hack = 1;

                break;

            case QEMU_OPTION_acpitable:

                do_acpitable_option(optarg);

                break;

            case QEMU_OPTION_smbios:

                do_smbios_option(optarg);

                break;

            case QEMU_OPTION_enable_kvm:

                olist = qemu_find_opts("machine");

                qemu_opts_reset(olist);

                qemu_opts_parse(olist, "accel=kvm", 0);

                break;

            case QEMU_OPTION_machine:

                olist = qemu_find_opts("machine");

                qemu_opts_reset(olist);

                opts = qemu_opts_parse(olist, optarg, 1);

                if (!opts) {

                    fprintf(stderr, "parse error: %s\n", optarg);

                    exit(1);

                }

                optarg = qemu_opt_get(opts, "type");

                if (optarg) {

                    machine = machine_parse(optarg);

                }

                break;

            case QEMU_OPTION_usb:

                usb_enabled = 1;

                break;

            case QEMU_OPTION_usbdevice:

                usb_enabled = 1;

                add_device_config(DEV_USB, optarg);

                break;

            case QEMU_OPTION_device:

                if (!qemu_opts_parse(qemu_find_opts("device"), optarg, 1)) {

                    exit(1);

                }

                break;

            case QEMU_OPTION_smp:

                smp_parse(optarg);

                if (smp_cpus < 1) {

                    fprintf(stderr, "Invalid number of CPUs\n");

                    exit(1);

                }

                if (max_cpus < smp_cpus) {

                    fprintf(stderr, "maxcpus must be equal to or greater than "

                            "smp\n");

                    exit(1);

                }

                if (max_cpus > 255) {

                    fprintf(stderr, "Unsupported number of maxcpus\n");

                    exit(1);

                }

                break;

	    case QEMU_OPTION_vnc:

#ifdef CONFIG_VNC

                display_remote++;

                vnc_display = optarg;

#else

                fprintf(stderr, "VNC support is disabled\n");

                exit(1);

#endif

                break;

            case QEMU_OPTION_no_acpi:

                acpi_enabled = 0;

                break;

            case QEMU_OPTION_no_hpet:

                no_hpet = 1;

                break;

            case QEMU_OPTION_balloon:

                if (balloon_parse(optarg) < 0) {

                    fprintf(stderr, "Unknown -balloon argument %s\n", optarg);

                    exit(1);

                }

                break;

            case QEMU_OPTION_no_reboot:

                no_reboot = 1;

                break;

            case QEMU_OPTION_no_shutdown:

                no_shutdown = 1;

                break;

            case QEMU_OPTION_show_cursor:

                cursor_hide = 0;

                break;

            case QEMU_OPTION_uuid:

                if(qemu_uuid_parse(optarg, qemu_uuid) < 0) {

                    fprintf(stderr, "Fail to parse UUID string."

                            " Wrong format.\n");

                    exit(1);

                }

                break;

	    case QEMU_OPTION_option_rom:

		if (nb_option_roms >= MAX_OPTION_ROMS) {

		    fprintf(stderr, "Too many option ROMs\n");

		    exit(1);

		}

                opts = qemu_opts_parse(qemu_find_opts("option-rom"), optarg, 1);

                option_rom[nb_option_roms].name = qemu_opt_get(opts, "romfile");

                option_rom[nb_option_roms].bootindex =

                    qemu_opt_get_number(opts, "bootindex", -1);

                if (!option_rom[nb_option_roms].name) {

                    fprintf(stderr, "Option ROM file is not specified\n");

                    exit(1);

                }

		nb_option_roms++;

		break;

            case QEMU_OPTION_semihosting:

                semihosting_enabled = 1;

                break;

            case QEMU_OPTION_name:

                qemu_name = g_strdup(optarg);

		 {

		     char *p = strchr(qemu_name, ',');

		     if (p != NULL) {

		        *p++ = 0;

			if (strncmp(p, "process=", 8)) {

			    fprintf(stderr, "Unknown subargument %s to -name\n", p);

			    exit(1);

			}

			p += 8;

			os_set_proc_name(p);

		     }	

		 }	

                break;

            case QEMU_OPTION_prom_env:

                if (nb_prom_envs >= MAX_PROM_ENVS) {

                    fprintf(stderr, "Too many prom variables\n");

                    exit(1);

                }

                prom_envs[nb_prom_envs] = optarg;

                nb_prom_envs++;

                break;

            case QEMU_OPTION_old_param:

                old_param = 1;

                break;

            case QEMU_OPTION_clock:

                configure_alarms(optarg);

                break;

            case QEMU_OPTION_startdate:

                configure_rtc_date_offset(optarg, 1);

                break;

            case QEMU_OPTION_rtc:

                opts = qemu_opts_parse(qemu_find_opts("rtc"), optarg, 0);

                if (!opts) {

                    exit(1);

                }

                configure_rtc(opts);

                break;

            case QEMU_OPTION_tb_size:

                tcg_tb_size = strtol(optarg, NULL, 0);

                if (tcg_tb_size < 0) {

                    tcg_tb_size = 0;

                }

                break;

            case QEMU_OPTION_icount:

                icount_option = optarg;

                break;

            case QEMU_OPTION_incoming:

                incoming = optarg;

                break;

            case QEMU_OPTION_nodefaults:

                default_serial = 0;

                default_parallel = 0;

                default_virtcon = 0;

                default_monitor = 0;

                default_vga = 0;

                default_net = 0;

                default_floppy = 0;

                default_cdrom = 0;

                default_sdcard = 0;

                break;

            case QEMU_OPTION_xen_domid:

                if (!(xen_available())) {

                    printf("Option %s not supported for this target\n", popt->name);

                    exit(1);

                }

                xen_domid = atoi(optarg);

                break;

            case QEMU_OPTION_xen_create:

                if (!(xen_available())) {

                    printf("Option %s not supported for this target\n", popt->name);

                    exit(1);

                }

                xen_mode = XEN_CREATE;

                break;

            case QEMU_OPTION_xen_attach:

                if (!(xen_available())) {

                    printf("Option %s not supported for this target\n", popt->name);

                    exit(1);

                }

                xen_mode = XEN_ATTACH;

                break;

            case QEMU_OPTION_trace:

            {

                opts = qemu_opts_parse(qemu_find_opts("trace"), optarg, 0);

                if (!opts) {

                    exit(1);

                }

                trace_events = qemu_opt_get(opts, "events");

                trace_file = qemu_opt_get(opts, "file");

                break;

            }

            case QEMU_OPTION_readconfig:

                {

                    int ret = qemu_read_config_file(optarg);

                    if (ret < 0) {

                        fprintf(stderr, "read config %s: %s\n", optarg,

                            strerror(-ret));

                        exit(1);

                    }

                    break;

                }

            case QEMU_OPTION_spice:

                olist = qemu_find_opts("spice");

                if (!olist) {

                    fprintf(stderr, "spice is not supported by this qemu build.\n");

                    exit(1);

                }

                opts = qemu_opts_parse(olist, optarg, 0);

                if (!opts) {

                    fprintf(stderr, "parse error: %s\n", optarg);

                    exit(1);

                }

                break;

            case QEMU_OPTION_writeconfig:

                {

                    FILE *fp;

                    if (strcmp(optarg, "-") == 0) {

                        fp = stdout;

                    } else {

                        fp = fopen(optarg, "w");

                        if (fp == NULL) {

                            fprintf(stderr, "open %s: %s\n", optarg, strerror(errno));

                            exit(1);

                        }

                    }

                    qemu_config_write(fp);

                    fclose(fp);

                    break;

                }

            default:

                os_parse_cmd_args(popt->index, optarg);

            }

        }

    }

    loc_set_none();



    /* Open the logfile at this point, if necessary. We can't open the logfile

     * when encountering either of the logging options (-d or -D) because the

     * other one may be encountered later on the command line, changing the

     * location or level of logging.

     */

    if (log_mask) {

        if (log_file) {

            set_cpu_log_filename(log_file);

        }

        set_cpu_log(log_mask);

    }



    if (!trace_backend_init(trace_events, trace_file)) {

        exit(1);

    }



    /* If no data_dir is specified then try to find it relative to the

       executable path.  */

    if (!data_dir) {

        data_dir = os_find_datadir(argv[0]);

    }

    /* If all else fails use the install path specified when building. */

    if (!data_dir) {

        data_dir = CONFIG_QEMU_DATADIR;

    }



    if (machine == NULL) {

        fprintf(stderr, "No machine found.\n");

        exit(1);

    }



    /*

     * Default to max_cpus = smp_cpus, in case the user doesn't

     * specify a max_cpus value.

     */

    if (!max_cpus)

        max_cpus = smp_cpus;



    machine->max_cpus = machine->max_cpus ?: 1; /* Default to UP */

    if (smp_cpus > machine->max_cpus) {

        fprintf(stderr, "Number of SMP cpus requested (%d), exceeds max cpus "

                "supported by machine `%s' (%d)\n", smp_cpus,  machine->name,

                machine->max_cpus);

        exit(1);

    }



    /*

     * Get the default machine options from the machine if it is not already

     * specified either by the configuration file or by the command line.

     */

    if (machine->default_machine_opts) {

        QemuOptsList *list = qemu_find_opts("machine");

        const char *p = NULL;



        if (!QTAILQ_EMPTY(&list->head)) {

            p = qemu_opt_get(QTAILQ_FIRST(&list->head), "accel");

        }

        if (p == NULL) {

            qemu_opts_reset(list);

            opts = qemu_opts_parse(list, machine->default_machine_opts, 0);

            if (!opts) {

                fprintf(stderr, "parse error for machine %s: %s\n",

                        machine->name, machine->default_machine_opts);

                exit(1);

            }

        }

    }



    qemu_opts_foreach(qemu_find_opts("device"), default_driver_check, NULL, 0);

    qemu_opts_foreach(qemu_find_opts("global"), default_driver_check, NULL, 0);



    if (machine->no_serial) {

        default_serial = 0;

    }

    if (machine->no_parallel) {

        default_parallel = 0;

    }

    if (!machine->use_virtcon) {

        default_virtcon = 0;

    }

    if (machine->no_vga) {

        default_vga = 0;

    }

    if (machine->no_floppy) {

        default_floppy = 0;

    }

    if (machine->no_cdrom) {

        default_cdrom = 0;

    }

    if (machine->no_sdcard) {

        default_sdcard = 0;

    }



    if (display_type == DT_NOGRAPHIC) {

        if (default_parallel)

            add_device_config(DEV_PARALLEL, "null");

        if (default_serial && default_monitor) {

            add_device_config(DEV_SERIAL, "mon:stdio");

        } else if (default_virtcon && default_monitor) {

            add_device_config(DEV_VIRTCON, "mon:stdio");

        } else {

            if (default_serial)

                add_device_config(DEV_SERIAL, "stdio");

            if (default_virtcon)

                add_device_config(DEV_VIRTCON, "stdio");

            if (default_monitor)

                monitor_parse("stdio", "readline");

        }

    } else {

        if (default_serial)

            add_device_config(DEV_SERIAL, "vc:80Cx24C");

        if (default_parallel)

            add_device_config(DEV_PARALLEL, "vc:80Cx24C");

        if (default_monitor)

            monitor_parse("vc:80Cx24C", "readline");

        if (default_virtcon)

            add_device_config(DEV_VIRTCON, "vc:80Cx24C");

    }

    if (default_vga)

        vga_interface_type = VGA_CIRRUS;



    socket_init();



    if (qemu_opts_foreach(qemu_find_opts("chardev"), chardev_init_func, NULL, 1) != 0)

        exit(1);

#ifdef CONFIG_VIRTFS

    if (qemu_opts_foreach(qemu_find_opts("fsdev"), fsdev_init_func, NULL, 1) != 0) {

        exit(1);

    }

#endif



    os_daemonize();



    if (pid_file && qemu_create_pidfile(pid_file) != 0) {

        os_pidfile_error();

        exit(1);

    }



    /* init the memory */

    if (ram_size == 0) {

        ram_size = DEFAULT_RAM_SIZE * 1024 * 1024;

    }



    configure_accelerator();



    qemu_init_cpu_loop();

    if (qemu_init_main_loop()) {

        fprintf(stderr, "qemu_init_main_loop failed\n");

        exit(1);

    }

    linux_boot = (kernel_filename != NULL);



    if (!linux_boot && *kernel_cmdline != '\0') {

        fprintf(stderr, "-append only allowed with -kernel option\n");

        exit(1);

    }



    if (!linux_boot && initrd_filename != NULL) {

        fprintf(stderr, "-initrd only allowed with -kernel option\n");

        exit(1);

    }



    os_set_line_buffering();



    if (init_timer_alarm() < 0) {

        fprintf(stderr, "could not initialize alarm timer\n");

        exit(1);

    }



    if (icount_option && (kvm_enabled() || xen_enabled())) {

        fprintf(stderr, "-icount is not allowed with kvm or xen\n");

        exit(1);

    }

    configure_icount(icount_option);



    if (net_init_clients() < 0) {

        exit(1);

    }



    /* init the bluetooth world */

    if (foreach_device_config(DEV_BT, bt_parse))

        exit(1);



    if (!xen_enabled()) {

        /* On 32-bit hosts, QEMU is limited by virtual address space */

        if (ram_size > (2047 << 20) && HOST_LONG_BITS == 32) {

            fprintf(stderr, "qemu: at most 2047 MB RAM can be simulated\n");

            exit(1);

        }

    }



    cpu_exec_init_all();



    bdrv_init_with_whitelist();



    blk_mig_init();



    /* open the virtual block devices */

    if (snapshot)

        qemu_opts_foreach(qemu_find_opts("drive"), drive_enable_snapshot, NULL, 0);

    if (qemu_opts_foreach(qemu_find_opts("drive"), drive_init_func, &machine->use_scsi, 1) != 0)

        exit(1);



    default_drive(default_cdrom, snapshot, machine->use_scsi,

                  IF_DEFAULT, 2, CDROM_OPTS);

    default_drive(default_floppy, snapshot, machine->use_scsi,

                  IF_FLOPPY, 0, FD_OPTS);

    default_drive(default_sdcard, snapshot, machine->use_scsi,

                  IF_SD, 0, SD_OPTS);



    register_savevm_live(NULL, "ram", 0, 4, NULL, ram_save_live, NULL,

                         ram_load, NULL);



    if (nb_numa_nodes > 0) {

        int i;



        if (nb_numa_nodes > MAX_NODES) {

            nb_numa_nodes = MAX_NODES;

        }



        /* If no memory size if given for any node, assume the default case

         * and distribute the available memory equally across all nodes

         */

        for (i = 0; i < nb_numa_nodes; i++) {

            if (node_mem[i] != 0)

                break;

        }

        if (i == nb_numa_nodes) {

            uint64_t usedmem = 0;



            /* On Linux, the each node's border has to be 8MB aligned,

             * the final node gets the rest.

             */

            for (i = 0; i < nb_numa_nodes - 1; i++) {

                node_mem[i] = (ram_size / nb_numa_nodes) & ~((1 << 23UL) - 1);

                usedmem += node_mem[i];

            }

            node_mem[i] = ram_size - usedmem;

        }



        for (i = 0; i < nb_numa_nodes; i++) {

            if (node_cpumask[i] != 0)

                break;

        }

        /* assigning the VCPUs round-robin is easier to implement, guest OSes

         * must cope with this anyway, because there are BIOSes out there in

         * real machines which also use this scheme.

         */

        if (i == nb_numa_nodes) {

            for (i = 0; i < max_cpus; i++) {

                node_cpumask[i % nb_numa_nodes] |= 1 << i;

            }

        }

    }



    if (qemu_opts_foreach(qemu_find_opts("mon"), mon_init_func, NULL, 1) != 0) {

        exit(1);

    }



    if (foreach_device_config(DEV_SERIAL, serial_parse) < 0)

        exit(1);

    if (foreach_device_config(DEV_PARALLEL, parallel_parse) < 0)

        exit(1);

    if (foreach_device_config(DEV_VIRTCON, virtcon_parse) < 0)

        exit(1);

    if (foreach_device_config(DEV_DEBUGCON, debugcon_parse) < 0)

        exit(1);



    module_call_init(MODULE_INIT_DEVICE);



    if (qemu_opts_foreach(qemu_find_opts("device"), device_help_func, NULL, 0) != 0)

        exit(0);



    if (watchdog) {

        i = select_watchdog(watchdog);

        if (i > 0)

            exit (i == 1 ? 1 : 0);

    }



    if (machine->compat_props) {

        qdev_prop_register_global_list(machine->compat_props);

    }

    qemu_add_globals();



    qdev_machine_init();



    machine->init(ram_size, boot_devices,

                  kernel_filename, kernel_cmdline, initrd_filename, cpu_model);



    cpu_synchronize_all_post_init();



    set_numa_modes();



    current_machine = machine;



    /* init USB devices */

    if (usb_enabled) {

        if (foreach_device_config(DEV_USB, usb_parse) < 0)

            exit(1);

    }



    /* init generic devices */

    if (qemu_opts_foreach(qemu_find_opts("device"), device_init_func, NULL, 1) != 0)

        exit(1);



    net_check_clients();



    /* just use the first displaystate for the moment */

    ds = get_displaystate();



    if (using_spice)

        display_remote++;

    if (display_type == DT_DEFAULT && !display_remote) {

#if defined(CONFIG_SDL) || defined(CONFIG_COCOA)

        display_type = DT_SDL;

#elif defined(CONFIG_VNC)

        vnc_display = "localhost:0,to=99";

        show_vnc_port = 1;

#else

        display_type = DT_NONE;

#endif

    }





    /* init local displays */

    switch (display_type) {

    case DT_NOGRAPHIC:

        break;

#if defined(CONFIG_CURSES)

    case DT_CURSES:

        curses_display_init(ds, full_screen);

        break;

#endif

#if defined(CONFIG_SDL)

    case DT_SDL:

        sdl_display_init(ds, full_screen, no_frame);

        break;

#elif defined(CONFIG_COCOA)

    case DT_SDL:

        cocoa_display_init(ds, full_screen);

        break;

#endif

    default:

        break;

    }



    /* must be after terminal init, SDL library changes signal handlers */

    os_setup_signal_handling();



#ifdef CONFIG_VNC

    /* init remote displays */

    if (vnc_display) {

        vnc_display_init(ds);

        if (vnc_display_open(ds, vnc_display) < 0)

            exit(1);



        if (show_vnc_port) {

            printf("VNC server running on `%s'\n", vnc_display_local_addr(ds));

        }

    }

#endif

#ifdef CONFIG_SPICE

    if (using_spice && !qxl_enabled) {

        qemu_spice_display_init(ds);

    }

#endif



    /* display setup */

    dpy_resize(ds);

    dcl = ds->listeners;

    while (dcl != NULL) {

        if (dcl->dpy_refresh != NULL) {

            ds->gui_timer = qemu_new_timer_ms(rt_clock, gui_update, ds);

            qemu_mod_timer(ds->gui_timer, qemu_get_clock_ms(rt_clock));

            break;

        }

        dcl = dcl->next;

    }

    text_consoles_set_display(ds);



    if (gdbstub_dev && gdbserver_start(gdbstub_dev) < 0) {

        fprintf(stderr, "qemu: could not open gdbserver on device '%s'\n",

                gdbstub_dev);

        exit(1);

    }



    qdev_machine_creation_done();



    if (rom_load_all() != 0) {

        fprintf(stderr, "rom loading failed\n");

        exit(1);

    }



    /* TODO: once all bus devices are qdevified, this should be done

     * when bus is created by qdev.c */

    qemu_register_reset(qbus_reset_all_fn, sysbus_get_default());

    qemu_run_machine_init_done_notifiers();



    qemu_system_reset(VMRESET_SILENT);

    if (loadvm) {

        if (load_vmstate(loadvm) < 0) {

            autostart = 0;

        }

    }



    if (incoming) {

        runstate_set(RUN_STATE_INMIGRATE);

        int ret = qemu_start_incoming_migration(incoming);

        if (ret < 0) {

            fprintf(stderr, "Migration failed. Exit code %s(%d), exiting.\n",

                    incoming, ret);

            exit(ret);

        }

    } else if (autostart) {

        vm_start();

    }



    os_setup_post();



    resume_all_vcpus();

    main_loop();

    bdrv_close_all();

    pause_all_vcpus();

    net_cleanup();

    res_free();



    return 0;

}
