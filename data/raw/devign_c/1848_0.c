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



    for (i = 0; i < 64; i++) {

        init_pair(i, colour_default[i & 7], colour_default[i >> 3]);

    }

    /* Set default color for more than 64. (monitor uses 0x74xx for example) */

    for (i = 64; i < COLOR_PAIRS; i++) {

        init_pair(i, COLOR_WHITE, COLOR_BLACK);

    }



    /*

     * Setup mapping for vga to curses line graphics.

     * FIXME: for better font, have to use ncursesw and setlocale()

     */

#if 0

    /* FIXME: map from where? */

    ACS_S1;

    ACS_S3;

    ACS_S7;

    ACS_S9;

#endif

    /* ACS_* is not constant. So, we can't initialize statically. */

    vga_to_curses['\0'] = ' ';

    vga_to_curses[0x04] = ACS_DIAMOND;

    vga_to_curses[0x0a] = ACS_RARROW;

    vga_to_curses[0x0b] = ACS_LARROW;

    vga_to_curses[0x18] = ACS_UARROW;

    vga_to_curses[0x19] = ACS_DARROW;

    vga_to_curses[0x9c] = ACS_STERLING;

    vga_to_curses[0xb0] = ACS_BOARD;

    vga_to_curses[0xb1] = ACS_CKBOARD;

    vga_to_curses[0xb3] = ACS_VLINE;

    vga_to_curses[0xb4] = ACS_RTEE;

    vga_to_curses[0xbf] = ACS_URCORNER;

    vga_to_curses[0xc0] = ACS_LLCORNER;

    vga_to_curses[0xc1] = ACS_BTEE;

    vga_to_curses[0xc2] = ACS_TTEE;

    vga_to_curses[0xc3] = ACS_LTEE;

    vga_to_curses[0xc4] = ACS_HLINE;

    vga_to_curses[0xc5] = ACS_PLUS;

    vga_to_curses[0xce] = ACS_LANTERN;

    vga_to_curses[0xd8] = ACS_NEQUAL;

    vga_to_curses[0xd9] = ACS_LRCORNER;

    vga_to_curses[0xda] = ACS_ULCORNER;

    vga_to_curses[0xdb] = ACS_BLOCK;

    vga_to_curses[0xe3] = ACS_PI;

    vga_to_curses[0xf1] = ACS_PLMINUS;

    vga_to_curses[0xf2] = ACS_GEQUAL;

    vga_to_curses[0xf3] = ACS_LEQUAL;

    vga_to_curses[0xf8] = ACS_DEGREE;

    vga_to_curses[0xfe] = ACS_BULLET;

}
