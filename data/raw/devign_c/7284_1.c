static void term_init(void)

{

#if HAVE_TERMIOS_H

    if(!run_as_daemon){

    struct termios tty;



    tcgetattr (0, &tty);

    oldtty = tty;

    atexit(term_exit);



    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP

                          |INLCR|IGNCR|ICRNL|IXON);

    tty.c_oflag |= OPOST;

    tty.c_lflag &= ~(ECHO|ECHONL|ICANON|IEXTEN);

    tty.c_cflag &= ~(CSIZE|PARENB);

    tty.c_cflag |= CS8;

    tty.c_cc[VMIN] = 1;

    tty.c_cc[VTIME] = 0;



    tcsetattr (0, TCSANOW, &tty);

    signal(SIGQUIT, sigterm_handler); /* Quit (POSIX).  */

    }

#endif

    avformat_network_deinit();



    signal(SIGINT , sigterm_handler); /* Interrupt (ANSI).    */

    signal(SIGTERM, sigterm_handler); /* Termination (ANSI).  */

#ifdef SIGXCPU

    signal(SIGXCPU, sigterm_handler);

#endif

}
