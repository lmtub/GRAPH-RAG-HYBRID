static int qemu_chr_open_win_stdio(QemuOpts *opts, CharDriverState **_chr)

{

    CharDriverState   *chr;

    WinStdioCharState *stdio;

    DWORD              dwMode;

    int                is_console = 0;



    if (stdio_nb_clients >= STDIO_MAX_CLIENTS

        || ((display_type != DT_NOGRAPHIC) && (stdio_nb_clients != 0))) {

        return -EIO;

    }



    chr   = g_malloc0(sizeof(CharDriverState));

    stdio = g_malloc0(sizeof(WinStdioCharState));



    stdio->hStdIn = GetStdHandle(STD_INPUT_HANDLE);

    if (stdio->hStdIn == INVALID_HANDLE_VALUE) {

        fprintf(stderr, "cannot open stdio: invalid handle\n");

        exit(1);

    }



    is_console = GetConsoleMode(stdio->hStdIn, &dwMode) != 0;



    chr->opaque    = stdio;

    chr->chr_write = win_stdio_write;

    chr->chr_close = win_stdio_close;



    if (stdio_nb_clients == 0) {

        if (is_console) {

            if (qemu_add_wait_object(stdio->hStdIn,

                                     win_stdio_wait_func, chr)) {

                fprintf(stderr, "qemu_add_wait_object: failed\n");

            }

        } else {

            DWORD   dwId;



            stdio->hInputReadyEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

            stdio->hInputDoneEvent  = CreateEvent(NULL, FALSE, FALSE, NULL);

            stdio->hInputThread     = CreateThread(NULL, 0, win_stdio_thread,

                                            chr, 0, &dwId);



            if (stdio->hInputThread == INVALID_HANDLE_VALUE

                || stdio->hInputReadyEvent == INVALID_HANDLE_VALUE

                || stdio->hInputDoneEvent == INVALID_HANDLE_VALUE) {

                fprintf(stderr, "cannot create stdio thread or event\n");

                exit(1);

            }

            if (qemu_add_wait_object(stdio->hInputReadyEvent,

                                     win_stdio_thread_wait_func, chr)) {

                fprintf(stderr, "qemu_add_wait_object: failed\n");

            }

        }

    }



    dwMode |= ENABLE_LINE_INPUT;



    stdio_clients[stdio_nb_clients++] = chr;

    if (stdio_nb_clients == 1 && is_console) {

        /* set the terminal in raw mode */

        /* ENABLE_QUICK_EDIT_MODE | ENABLE_EXTENDED_FLAGS */

        dwMode |= ENABLE_PROCESSED_INPUT;

    }



    SetConsoleMode(stdio->hStdIn, dwMode);



    chr->chr_set_echo = qemu_chr_set_echo_win_stdio;

    qemu_chr_fe_set_echo(chr, false);



    *_chr = chr;



    return 0;

}
