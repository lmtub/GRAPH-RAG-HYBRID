static void tty_serial_init(int fd, int speed,

                            int parity, int data_bits, int stop_bits)

{

    struct termios tty;

    speed_t spd;



#if 0

    printf("tty_serial_init: speed=%d parity=%c data=%d stop=%d\n",

           speed, parity, data_bits, stop_bits);

#endif

    tcgetattr (fd, &tty);

    oldtty = tty;



#define check_speed(val) if (speed <= val) { spd = B##val; break; }

    speed = speed * 10 / 11;

    do {

        check_speed(50);

        check_speed(75);

        check_speed(110);

        check_speed(134);

        check_speed(150);

        check_speed(200);

        check_speed(300);

        check_speed(600);

        check_speed(1200);

        check_speed(1800);

        check_speed(2400);

        check_speed(4800);

        check_speed(9600);

        check_speed(19200);

        check_speed(38400);

        /* Non-Posix values follow. They may be unsupported on some systems. */

        check_speed(57600);

        check_speed(115200);

#ifdef B230400

        check_speed(230400);

#endif

#ifdef B460800

        check_speed(460800);

#endif

#ifdef B500000

        check_speed(500000);

#endif

#ifdef B576000

        check_speed(576000);

#endif

#ifdef B921600

        check_speed(921600);

#endif

#ifdef B1000000

        check_speed(1000000);

#endif

#ifdef B1152000

        check_speed(1152000);

#endif

#ifdef B1500000

        check_speed(1500000);

#endif

#ifdef B2000000

        check_speed(2000000);

#endif

#ifdef B2500000

        check_speed(2500000);

#endif

#ifdef B3000000

        check_speed(3000000);

#endif

#ifdef B3500000

        check_speed(3500000);

#endif

#ifdef B4000000

        check_speed(4000000);

#endif

        spd = B115200;

    } while (0);



    cfsetispeed(&tty, spd);

    cfsetospeed(&tty, spd);



    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP

                          |INLCR|IGNCR|ICRNL|IXON);

    tty.c_oflag |= OPOST;

    tty.c_lflag &= ~(ECHO|ECHONL|ICANON|IEXTEN|ISIG);

    tty.c_cflag &= ~(CSIZE|PARENB|PARODD|CRTSCTS|CSTOPB);

    switch(data_bits) {

    default:

    case 8:

        tty.c_cflag |= CS8;

        break;

    case 7:

        tty.c_cflag |= CS7;

        break;

    case 6:

        tty.c_cflag |= CS6;

        break;

    case 5:

        tty.c_cflag |= CS5;

        break;

    }

    switch(parity) {

    default:

    case 'N':

        break;

    case 'E':

        tty.c_cflag |= PARENB;

        break;

    case 'O':

        tty.c_cflag |= PARENB | PARODD;

        break;

    }

    if (stop_bits == 2)

        tty.c_cflag |= CSTOPB;



    tcsetattr (fd, TCSANOW, &tty);

}
