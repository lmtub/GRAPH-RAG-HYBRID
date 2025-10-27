static void term_exit(void)

{

    tcsetattr(0, TCSANOW, &oldtty);

}
