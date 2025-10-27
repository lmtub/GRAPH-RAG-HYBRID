main(

    int argc,

    char *argv[]

) {

    GMainLoop *loop;

    GIOChannel *channel_stdin;

    char *qemu_host;

    char *qemu_port;

    VSCMsgHeader mhHeader;



    VCardEmulOptions *command_line_options = NULL;



    char *cert_names[MAX_CERTS];

    char *emul_args = NULL;

    int cert_count = 0;

    int c, sock;



    if (socket_init() != 0)

        return 1;



    while ((c = getopt(argc, argv, "c:e:pd:")) != -1) {

        switch (c) {

        case 'c':

            if (cert_count >= MAX_CERTS) {

                printf("too many certificates (max = %d)\n", MAX_CERTS);

                exit(5);

            }

            cert_names[cert_count++] = optarg;

            break;

        case 'e':

            emul_args = optarg;

            break;

        case 'p':

            print_usage();

            exit(4);

            break;

        case 'd':

            verbose = get_id_from_string(optarg, 1);

            break;

        }

    }



    if (argc - optind != 2) {

        print_usage();

        exit(4);

    }



    if (cert_count > 0) {

        char *new_args;

        int len, i;

        /* if we've given some -c options, we clearly we want do so some

         * software emulation.  add that emulation now. this is NSS Emulator

         * specific */

        if (emul_args == NULL) {

            emul_args = (char *)"db=\"/etc/pki/nssdb\"";

        }

#define SOFT_STRING ",soft=(,Virtual Reader,CAC,,"

             /* 2 == close paren & null */

        len = strlen(emul_args) + strlen(SOFT_STRING) + 2;

        for (i = 0; i < cert_count; i++) {

            len += strlen(cert_names[i])+1; /* 1 == comma */

        }

        new_args = g_malloc(len);

        strcpy(new_args, emul_args);

        strcat(new_args, SOFT_STRING);

        for (i = 0; i < cert_count; i++) {

            strcat(new_args, cert_names[i]);

            strcat(new_args, ",");

        }

        strcat(new_args, ")");

        emul_args = new_args;

    }

    if (emul_args) {

        command_line_options = vcard_emul_options(emul_args);

    }



    qemu_host = g_strdup(argv[argc - 2]);

    qemu_port = g_strdup(argv[argc - 1]);

    sock = connect_to_qemu(qemu_host, qemu_port);

    if (sock == -1) {

        fprintf(stderr, "error opening socket, exiting.\n");

        exit(5);

    }



    socket_to_send = g_byte_array_new();

    qemu_mutex_init(&socket_to_send_lock);

    qemu_mutex_init(&pending_reader_lock);

    qemu_cond_init(&pending_reader_condition);



    vcard_emul_init(command_line_options);



    loop = g_main_loop_new(NULL, true);



    printf("> ");

    fflush(stdout);



#ifdef _WIN32

    channel_stdin = g_io_channel_win32_new_fd(STDIN_FILENO);

#else

    channel_stdin = g_io_channel_unix_new(STDIN_FILENO);

#endif

    g_io_add_watch(channel_stdin, G_IO_IN, do_command, NULL);

#ifdef _WIN32

    channel_socket = g_io_channel_win32_new_socket(sock);

#else

    channel_socket = g_io_channel_unix_new(sock);

#endif

    g_io_channel_set_encoding(channel_socket, NULL, NULL);

    /* we buffer ourself for thread safety reasons */

    g_io_channel_set_buffered(channel_socket, FALSE);



    /* Send init message, Host responds (and then we send reader attachments) */

    VSCMsgInit init = {

        .version = htonl(VSCARD_VERSION),

        .magic = VSCARD_MAGIC,

        .capabilities = {0}

    };

    send_msg(VSC_Init, mhHeader.reader_id, &init, sizeof(init));



    g_main_loop_run(loop);

    g_main_loop_unref(loop);



    g_io_channel_unref(channel_stdin);

    g_io_channel_unref(channel_socket);

    g_byte_array_unref(socket_to_send);



    closesocket(sock);

    return 0;

}
