static void curses_setup(void)

{

    int i, colour_default[8] = {

        COLOR_BLACK, COLOR_BLUE, COLOR_GREEN, COLOR_CYAN,

        COLOR_RED, COLOR_MAGENTA, COLOR_YELLOW, COLOR_WHITE,

    };



    /* input as raw as possible, let everything be interpreted

     * by the guest system */

    initscr(); noecho(); intrflush(stdscr, FALSE);

    nodelay(stdscr, TRUE); nonl(); keypad(stdscr, TRUE);

    start_color(); raw(); scrollok(stdscr, FALSE);



    for (i = 0; i < 64; i ++)

        init_pair(i, colour_default[i & 7], colour_default[i >> 3]);

}
