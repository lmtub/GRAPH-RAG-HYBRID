static int read_password(char *buf, int buf_size)

{

    int c, i;

    printf("Password: ");

    fflush(stdout);

    i = 0;

    for(;;) {

        c = getchar();

        if (c == '\n')

            break;

        if (i < (buf_size - 1))

            buf[i++] = c;

    }

    buf[i] = '\0';

    return 0;

}
