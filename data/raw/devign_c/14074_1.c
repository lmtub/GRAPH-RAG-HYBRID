static int read_key(void)

{

#if HAVE_TERMIOS_H

    int n = 1;

    unsigned char ch;

    struct timeval tv;

    fd_set rfds;



    FD_ZERO(&rfds);

    FD_SET(0, &rfds);

    tv.tv_sec = 0;

    tv.tv_usec = 0;

    n = select(1, &rfds, NULL, NULL, &tv);

    if (n > 0) {

        n = read(0, &ch, 1);

        if (n == 1)

            return ch;



        return n;

    }

#elif HAVE_CONIO_H

    if(kbhit())

        return(getch());

#endif

    return -1;

}
